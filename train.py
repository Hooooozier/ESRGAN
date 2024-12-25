import torch
import config
from torch import nn
from torch import optim
from utils import gradient_penalty, load_checkpoint, save_checkpoint, plot_examples, initialize_weights_kaiming
from loss import VGGLoss
from torch.utils.data import DataLoader
from esrgan import Generator, Discriminator, disc_config
from tqdm import tqdm
# from dataset import DIV2K_DS
from dataset2 import DIV2K_DS
from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.benchmark = True

def train_fn(
    loader,
    disc,
    gen,
    opt_gen,
    opt_disc,
    vgg_loss,
    g_scaler,
    d_scaler,
    writer,
    tb_step,
    epoch
):
    l1 = nn.L1Loss()
    loop = tqdm(loader, desc=f"Epoch:{epoch}", leave=True)

    for idx, (low_res, high_res) in enumerate(loop):
        high_res = high_res.to(config.DEVICE)
        low_res = low_res.to(config.DEVICE)

        # train discriminator
        # with torch.amp.autocast(config.DEVICE):
        with torch.cuda.amp.autocast():
            fake = gen(low_res)
            score_real = disc(high_res)
            score_fake = disc(fake.detach())
            gp = gradient_penalty(disc, high_res, fake, device=config.DEVICE)
            loss_disc = config.LAMBDA_GP * gp -(torch.mean(score_real) - torch.mean(score_fake))

        opt_disc.zero_grad()
        d_scaler.scale(loss_disc).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # train generator
        # with torch.amp.autocast(config.DEVICE):
        with torch.cuda.amp.autocast():
            l1_loss = 1e-2 * l1(fake, high_res)
            adversarial_loss = 5e-3 * -torch.mean(disc(fake))
            loss_for_vgg = vgg_loss(fake, high_res)
            gen_loss = l1_loss + loss_for_vgg + adversarial_loss

        opt_gen.zero_grad()
        g_scaler.scale(gen_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        writer.add_scalar("Discriminator loss", loss_disc.item(), global_step=tb_step)
        writer.add_scalar("Generator loss", gen_loss.item(), global_step=tb_step)
        tb_step += 1

        loop.set_postfix(
            gp=gp.item(),
            critic=loss_disc.item(),
            l1=l1_loss.item(),
            vgg=loss_for_vgg.item(),
            adversarial=adversarial_loss.item(),
        )

    return tb_step


def main():
    dataset = DIV2K_DS(config.TRAIN_DATASET_PATH)
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
    )

    import os
    if not os.path.exists(config.VALID_SAVE_PATH):
        os.makedirs(config.VALID_SAVE_PATH)

    gen = Generator(config.IMG_CHANNELS).to(config.DEVICE)
    disc = Discriminator(disc_config=disc_config).to(config.DEVICE)

    # use kaiming initialize for ESRGAN, spectral for semantic gan(TODO Later )
    initialize_weights_kaiming(gen)

    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.9))

    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.9))
    writer = SummaryWriter(os.path.join(config.EXP_NAME, "logs_sunrgbd"))

    tb_step = 0
    gen.train()
    disc.train()
    vgg_loss = VGGLoss("ESRGAN")

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    # g_scaler = torch.amp.GradScaler()
    # d_scaler = torch.amp.GradScaler()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN,
            gen,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC,
            disc,
            opt_disc,
            config.LEARNING_RATE,
        )


    for epoch in range(config.NUM_EPOCHS):
        tb_step = train_fn(
            loader,
            disc,
            gen,
            opt_gen,
            opt_disc,
            vgg_loss,
            g_scaler,
            d_scaler,
            writer,
            tb_step,
            epoch
        )

        plot_examples(config.VALID_DATASET_PATH, gen)

        if config.SAVE_MODEL:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)


if __name__ == "__main__":
    main()
