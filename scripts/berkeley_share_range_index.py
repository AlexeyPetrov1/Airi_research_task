"""Bounded exploratory tar-name recovery from PNG-heavy split gzip byte ranges.

For indexing ONLY: restart at an outer DEFLATE stored-block boundary with an
unknown 32 KiB dictionary replaced by zeros. Reject the first 32 KiB of output
and require valid tar header checksums. No recovered image is accepted here.
The gzip stream CRC cannot be checked from a partial range. This is not a
replacement for image CRC + pixel verification or an official archive index.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote
import argparse,hashlib,io,json,re,tarfile,time,zlib
import requests
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/berkeley_ur5_improvement_v1/sources'
MIB=1024**2


def recover(data):
    if data.startswith(b'\x1f\x8b'):
        return zlib.decompressobj(31).decompress(data,16*MIB),dict(mode='gzip_from_header',compressed_offset=0)
    # The LEN/NLEN pair identifies byte-aligned payloads of stored blocks.
    # A fresh stored-block header is inserted before the original length pair.
    for offset in range(min(len(data)-4,256*1024)):
        length=int.from_bytes(data[offset:offset+2],'little')
        inverse=int.from_bytes(data[offset+2:offset+4],'little')
        if length < 4096 or length^inverse != 65535:continue
        try:
            decoder=zlib.decompressobj(-15,zdict=bytes(32768))
            output=decoder.decompress(b'\0'+data[offset:],16*MIB)
        except zlib.error:continue
        if valid_start(output) is not None:
            return output,dict(mode='raw_stored_block_unknown_dictionary',compressed_offset=offset,
                synthetic_header_bytes=1,unknown_dictionary_bytes=32768)
    # Most outer blocks can be dynamic Huffman blocks rather than stored ones.
    # Scan bounded bit positions; a wrong candidate must fail inflate or tar
    # checksum verification. Keep this exploratory and never accept pixels.
    array=np.frombuffer(data,np.uint8)
    for bit in range(8):
        shifted=data if bit==0 else ((array[:-1]>>bit)|(array[1:]<<(8-bit))).tobytes()
        for offset in range(min(len(shifted)-1024,128*1024)):
            if shifted[offset]&7 not in (2,4):continue  # non-final fixed/dynamic Huffman header
            try:
                decoder=zlib.decompressobj(-15,zdict=bytes(32768))
                first=decoder.decompress(shifted[offset:offset+1024],MIB)
                if len(first)<64:continue
                output=first+decoder.decompress(decoder.unconsumed_tail+shifted[offset+1024:],16*MIB-len(first))
            except zlib.error:continue
            if valid_start(output) is not None:
                return output,dict(mode='raw_huffman_block_unknown_dictionary',compressed_offset=offset,
                    compressed_bit_offset=bit,unknown_dictionary_bytes=32768)
    raise ValueError('No checksum-valid tar headers found after stored-block restart')


def valid_start(data):
    for match in re.finditer(b'ustar',data):
        begin=match.start()-257
        if begin<32768 or begin+512>len(data):continue
        block=data[begin:begin+512]
        try:checksum=int(block[148:156].strip(b'\0 '),8)
        except ValueError:continue
        if sum(block[:148])+8*32+sum(block[156:])==checksum:return begin


def list_members(data):
    begin=valid_start(data)
    if begin is None:return [],None
    entries=[];error=None
    try:
        with tarfile.open(fileobj=io.BytesIO(data[begin:]),mode='r|') as archive:
            for member in archive:
                entries.append(dict(path=member.name,size=member.size,
                    recovered_uncompressed_header_offset=begin+member.offset))
                if len(entries)>=32:break
    except (tarfile.TarError,EOFError) as e:error=repr(e)
    return entries,error


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parts',nargs='+',default=['aa','ab','au','cl'])
    p.add_argument('--offset',type=int,default=0);p.add_argument('--mib',type=int,default=4);a=p.parse_args()
    if a.mib>16 or len(a.parts)>64:raise ValueError('Bounded ranges only')
    rev=json.loads((OUT/'sharerobot_metadata.json').read_text())['sha'];start=time.monotonic()
    def probe(suffix):
        path='planning/images/rt_frames_success.tar.gz.part.'+suffix
        url=f'https://huggingface.co/datasets/BAAI/ShareRobot/resolve/{rev}/'+quote(path,safe='/')
        row=dict(path=path,url=url,requested_offset=a.offset,requested_bytes=a.mib*MIB)
        try:
            with requests.get(url,headers={'Range':f'bytes={a.offset}-{a.offset+a.mib*MIB-1}'},
                              stream=True,timeout=(15,40)) as r:
                r.raise_for_status()
                if r.status_code!=206:raise ValueError('Range request not respected')
                data=r.raw.read(a.mib*MIB)
                row.update(content_range=r.headers.get('Content-Range'),bytes_read=len(data),
                    compressed_bytes_sha256=hashlib.sha256(data).hexdigest())
            output,restart=recover(data);entries,error=list_members(output)
            row.update(restart=restart,recovered_output_bytes=len(output),entries=entries,partial_tar_error=error)
        except Exception as e:row['error']=repr(e)
        print(json.dumps(dict(part=suffix,entries=len(row.get('entries',[])),
            first_paths=[x['path'] for x in row.get('entries',[])[:2]],error=row.get('error'))),flush=True)
        return row
    with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(probe,a.parts))
    dest=OUT/f'share_range_names_{a.offset}_{a.parts[0]}_{a.parts[-1]}.json'
    dest.write_text(json.dumps(dict(revision=rev,elapsed_s=time.monotonic()-start,probes=rows,
        image_mapping_confirmed=False,exploratory_only=True,
        caveat='Checksum-valid tar names after partial DEFLATE restart; no complete-stream CRC and no images accepted.'),
        ensure_ascii=False,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':main()
