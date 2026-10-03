"""Integrity checks for accepting real image evidence from a partial archive."""
import gzip,io,sys,tarfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from berkeley_share_range_index import recover,valid_start
from berkeley_share_extract_range import check_png


class RangeEvidenceTests(unittest.TestCase):
    def image(self,seed):
        pixels=np.random.default_rng(seed).integers(0,256,(480,640,3),dtype=np.uint8)
        output=io.BytesIO();Image.fromarray(pixels).save(output,format='PNG')
        return output.getvalue()

    def test_every_png_chunk_crc_rejects_corrupted_payload(self):
        png=self.image(1);self.assertGreater(check_png(png),2)
        damaged=bytearray(png);pos=png.index(b'IDAT')+10;damaged[pos]^=1
        with self.assertRaisesRegex(ValueError,'CRC'):check_png(bytes(damaged))

    def test_truncated_image_is_not_accepted(self):
        with self.assertRaisesRegex(ValueError,'Truncated'):check_png(self.image(2)[:-8])

    def test_partial_stored_block_recovery_preserves_later_images_exactly(self):
        originals={f'images/episode_9/frame_{i}.png':self.image(i+4) for i in range(3)}
        raw=io.BytesIO()
        with tarfile.open(fileobj=raw,mode='w') as archive:
            for name,data in originals.items():
                info=tarfile.TarInfo(name);info.size=len(data);archive.addfile(info,io.BytesIO(data))
        compressed=gzip.compress(raw.getvalue(),compresslevel=0)
        output,receipt=recover(compressed[400000:])
        self.assertEqual(receipt['mode'],'raw_stored_block_unknown_dictionary')
        begin=valid_start(output)
        recovered={}
        with tarfile.open(fileobj=io.BytesIO(output[begin:]),mode='r|') as archive:
            for member in archive:
                data=archive.extractfile(member).read();check_png(data);recovered[member.name]=data
        self.assertEqual(set(recovered),set(list(originals)[1:]))
        for name,data in recovered.items():self.assertEqual(data,originals[name])


if __name__=='__main__':unittest.main()
