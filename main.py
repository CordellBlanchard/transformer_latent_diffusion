import argparse

from tld.train_distill import main
from tld.configs import DataConfig, ModelConfig, TrainConfig
import wandb

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--latent-path', type = str, default = '../image_latents_1m.npy')
    argparser.add_argument('--text-emb-path', type = str, default = '../text_encodings_1m.npy')
    argparser.add_argument('--val-enc-path', type = str, default = '../val_encs.npy')
    argparser.add_argument('--teacher-model', type = str, default = '../state_dict_378000.pth')
    argparser.add_argument('--from-checkpoint', action = 'store_true') # Default is from scratch
    argparser.add_argument('--run-id', type = str, default = None) # Remark: run_id being none avoids restoring from wandb, using a local model file instead.
    argparser.add_argument('--model-name', type = str, default = "full_state_dict.pth")
    argparser.add_argument('--checkpoint-model-name', type = str, default = None)
    argparser.add_argument('--disable-individual-checkpoints', action = 'store_true')
    argparser.add_argument('--batch-size', type = int, default = 64)
    argparser.add_argument('--lr', type=float, default=3e-4)
    argparser.add_argument('--wandb-mode', type = str, choices = ['online', 'offline'], default = 'online')
    args = argparser.parse_args()

    # Remark: accelerate might be doing its own tracking shit under the hood, we don't really care
    wandb.init(project = "transformer-latent-diffusion", mode = args.wandb_mode)

    data_config = DataConfig(
        latent_path = args.latent_path,
        text_emb_path = args.text_emb_path,
        val_path = args.val_enc_path
    )
    train_config = TrainConfig(
        compile = False,  # torch.compile not supported on Windows
        batch_size = args.batch_size,
        lr=args.lr,
        from_scratch = not args.from_checkpoint,
        model_name = args.model_name,
        checkpoint_model_name = args.checkpoint_model_name,
        save_individual_checkpoints = not args.disable_individual_checkpoints,
        run_id = args.run_id,
        teacher_model_name = args.teacher_model,
    )
    model_config = ModelConfig(data_config = data_config, train_config = train_config)

    main(model_config)

    wandb.finish()