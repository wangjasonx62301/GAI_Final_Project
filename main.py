from model.encode_decode_model import *
from model.text_model import *
from utils.config import *
from scripts.image_text_gen import *
from scripts.train_encoder import *

if __name__ == '__main__':
    
    encoder_checkpoint = 'checkpoint\\encoder.pt'
    decoder_checkpoint = 'checkpoint\\decoder.pt'
    EncoderDecoder_checkpoint = 'checkpoint\\text_gen_train_for_100_iters.pt'
    
    # load encoder
    if encoder_checkpoint == None:    
        # training encoder
        encoder = Encoder(config).to(config.device)
        train_encoder(encoder)
    else : 
        encoder = Encoder(config).to(config.device)
        load_checkpoint(encoder, encoder_checkpoint)
    print('=======================================')
    print('-       load encoder complete         -')
    print('=======================================')
    # load decoder
    if decoder_checkpoint == None:
        # training decoder
        decoder = Decoder(config).to(config.device)
        optim = torch.optim.AdamW(decoder.parameters(), lr=config.lr)
        train_gen(decoder, optim)
    
    else:
        decoder = Decoder(config).to(config.device)
        load_checkpoint(decoder, decoder_checkpoint)
    print('=======================================')
    print('-       load decoder complete         -')
    print('=======================================')
    # load EncoderDecoder
    if EncoderDecoder_checkpoint == None:
        model = EncoderDecoderModel(encoder=encoder, decoder=decoder, config=config).to(config.device)
        optim = torch.optim.AdamW(model.parameters(), lr=config.lr)
        train_image_text(model, optim, config.train_df, config.valid_df, config.train_image_dir, config.valid_image_dir)
    else :
        model = EncoderDecoderModel(encoder=encoder, decoder=decoder, config=config).to(config.device)
        load_checkpoint(model, EncoderDecoder_checkpoint)
    print('=======================================')
    print('-    load EncoderDecoder complete     -')
    print('=======================================')
    # make submission.csv
    get_submission(model)        


    
    
    
    