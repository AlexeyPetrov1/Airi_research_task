"""Extract only exact Berkeley targets from a bounded, pinned gzip range.

Partial DEFLATE restart is exploratory. Accept PNGs only after tar-header
validation, every PNG chunk CRC, Pillow verification and exact 640x480 size.
Pixel mapping is a separate subsequent check. Never writes the large archive.
"""
from pathlib import Path
import argparse,hashlib,io,json,re,struct,tarfile,time,zlib
from urllib.parse import quote
import numpy as np
from PIL import Image
import requests
from berkeley_share_range_index import recover,valid_start,MIB,OUT


def check_png(data):
    if data[:8]!=b'\x89PNG\r\n\x1a\n':raise ValueError('Not PNG')
    pos=8;chunks=0;ended=False
    while pos<len(data):
        if pos+12>len(data):raise ValueError('Truncated PNG chunk')
        length=struct.unpack('>I',data[pos:pos+4])[0]
        tag=data[pos+4:pos+8];end=pos+8+length
        if end+4>len(data):raise ValueError('Truncated PNG chunk')
        payload=data[pos+4:end]
        checksum=struct.unpack('>I',data[end:end+4])[0]
        if zlib.crc32(payload)!=checksum:raise ValueError('PNG chunk CRC mismatch')
        pos=end+4;chunks+=1
        if tag==b'IEND':ended=True;break
    if not ended or pos!=len(data):raise ValueError('Truncated/trailing PNG')
    im=Image.open(io.BytesIO(data))
    if im.size!=(640,480):raise ValueError('Wrong source image size')
    im.verify()
    return chunks


class RestartStream:
    def __init__(self,response,initial,restart):
        self.response=response;self.compressed_bytes=len(initial)
        self.bit=restart.get('compressed_bit_offset',0);self.tail=b''
        self.decoder=zlib.decompressobj(-15,zdict=bytes(32768))
        offset=restart['compressed_offset']
        payload=self.shift(initial[offset:])
        if restart['mode']=='raw_stored_block_unknown_dictionary':payload=b'\0'+payload
        elif restart['mode']=='gzip_from_header':
            self.decoder=zlib.decompressobj(31);payload=initial
        output=self.decoder.decompress(payload)
        begin=valid_start(output)
        if begin is None:raise ValueError('Could not reproduce tar restart')
        self.buffer=output[begin:];self.position=0;self.recovered_bytes=len(output)

    def shift(self,data):
        if not self.bit:return data
        data=self.tail+data;self.tail=data[-1:]
        array=np.frombuffer(data,np.uint8)
        return ((array[:-1]>>self.bit)|(array[1:]<<(8-self.bit))).tobytes()

    def read(self,size=-1):
        if size<0:raise ValueError('Streaming reads must be bounded')
        chunks=[];remaining=size
        while remaining:
            available=len(self.buffer)-self.position
            if available:
                take=min(available,remaining)
                chunks.append(self.buffer[self.position:self.position+take]);self.position+=take;remaining-=take
            else:
                data=self.response.raw.read(MIB)
                if not data:break
                self.compressed_bytes+=len(data)
                self.buffer=self.decoder.decompress(self.shift(data));self.position=0
                self.recovered_bytes+=len(self.buffer)
        return b''.join(chunks)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--part',required=True)
    p.add_argument('--start',type=int,default=0);p.add_argument('--stop',type=int,default=5368709120);a=p.parse_args()
    if a.stop-a.start>5368709120 or a.stop<=a.start:raise ValueError('At most one archive part')
    rev=json.loads((OUT/'sharerobot_metadata.json').read_text())['sha']
    path='planning/images/rt_frames_success.tar.gz.part.'+a.part
    url=f'https://huggingface.co/datasets/BAAI/ShareRobot/resolve/{rev}/'+quote(path,safe='/')
    dest=OUT/'share_exact_frames';dest.mkdir(exist_ok=True)
    receipt=OUT/f'share_extract_{a.part}_{a.start}.json';start=time.monotonic();last_print=start
    record=dict(revision=rev,source=url,requested_range=[a.start,a.stop-1],state='running',
        accepted_images=[],member_count=0,whole_gzip_crc_checked=False,pixel_mapping_confirmed=False)
    def save():receipt.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    known_frames={ep:{p.name for p in (dest/f'episode_{ep}').glob('frame_*.png')} for ep in (9,10)}
    save()
    try:
        with requests.get(url,headers={'Range':f'bytes={a.start}-{a.stop-1}'},stream=True,timeout=(20,90)) as response:
            response.raise_for_status()
            if response.status_code!=206:raise ValueError('Range ignored')
            record['content_range']=response.headers.get('Content-Range')
            first=response.raw.read(4*MIB);_,restart=recover(first)
            record['restart']=restart;record['initial_range_sha256']=hashlib.sha256(first).hexdigest()
            stream=RestartStream(response,first,restart)
            with tarfile.open(fileobj=stream,mode='r|') as archive:
                for member in archive:
                    record['member_count']+=1;record['last_member']=member.name
                    match=re.search(r'/43_berkeley_autolab_ur5#episode_(9|10)/frame_(\d+)\.png$',member.name)
                    if match:
                        ep,frame=map(int,match.groups())
                        if not 0<=frame<30:raise ValueError('Unexpected planning frame index')
                        data=archive.extractfile(member).read();crc_chunks=check_png(data)
                        saved=dest/f'episode_{ep}'/f'frame_{frame:02d}.png';saved.parent.mkdir(exist_ok=True)
                        if saved.exists() and saved.read_bytes()!=data:raise ValueError('Prior target bytes differ')
                        saved.write_bytes(data)
                        item=dict(member=member.name,episode=ep,frame=frame,bytes=len(data),
                            sha256=hashlib.sha256(data).hexdigest(),png_chunks_crc_checked=crc_chunks,
                            image_size=[640,480],local_path=str(saved.relative_to(OUT)),
                            compressed_position_upper_bound=a.start+stream.compressed_bytes)
                        record['accepted_images'].append(item);print(json.dumps(dict(accepted=item)),flush=True);save()
                        known_frames[ep].add(saved.name)
                    record.update(compressed_bytes_read=stream.compressed_bytes,elapsed_s=time.monotonic()-start)
                    if time.monotonic()-last_print>20:
                        print(json.dumps(dict(part=a.part,compressed_GiB=stream.compressed_bytes/1024**3,
                            member_count=record['member_count'],last_member=member.name,
                            accepted=len(record['accepted_images']),elapsed_s=record['elapsed_s'])),flush=True)
                        save();last_print=time.monotonic()
                    counts=[len(known_frames[ep]) for ep in (9,10)]
                    if counts==[30,30]:record['state']='all_target_pngs_extracted';break
                else:record['state']='range_finished'
    except Exception as e:record.update(state='partial_range_or_error',error=repr(e))
    record['elapsed_s']=time.monotonic()-start;save()
    print(json.dumps(dict(state=record['state'],accepted=len(record['accepted_images']),
        bytes_read=record.get('compressed_bytes_read'),error=record.get('error'))),flush=True)


if __name__=='__main__':main()
