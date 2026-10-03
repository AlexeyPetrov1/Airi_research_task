"""Verify native model bytes against the pinned Hub LFS SHA-256 receipts."""
from fmb_v2_scene import ROOT,RUN
from berkeley_preprocess import sha256,write_json

def main():
    results={}
    for name in ['MolmoMotion-4B-H3-F30','MolmoMotion-4B-H1-F32']:
        checkpoint=ROOT/'data/checkpoints'/name
        revision,expected,*_= (checkpoint/'.cache/huggingface/download/model.pt.metadata').read_text().splitlines()
        actual=sha256(checkpoint/'model.pt')
        assert actual==expected,f'Checkpoint corruption: {name}'
        results[name]={'revision':revision,'expected_LFS_sha256':expected,'actual_sha256':actual,
                       'model_bytes':(checkpoint/'model.pt').stat().st_size,'success':True}
        print(name,'SHA256 PASS',flush=True)
    write_json(RUN/'checkpoint_byte_audit.json',results)

if __name__=='__main__':main()
