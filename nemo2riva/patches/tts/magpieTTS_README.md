# Decoder  
Command to export decoder:  
`nemo2riva <model.ckpt path> --load_ckpt --model_config <hparams_file> --audio_codecpath <codec .nemo ckpt> --key tlt_encode --out magpie_decoder.riva --submodel decoder`  

# Encoder  
Command to export encoder:  
`nemo2riva <model.ckpt path> --load_ckpt --model_config <hparams_file> --audio_codecpath <codec .nemo ckpt> --key tlt_encode --out magpie_encoder.riva --submodel encoder`  
