from models import *
from data import *
from utils import *
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import time
import datetime
import os
import PIL.Image as Image
from apex import amp

# device = (1 if torch.cuda.is_available() else 'cpu')
seed = 1
torch.manual_seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
np.random.seed(seed)




num_epochs = 500
batch_size = 16
learning_rate = 0.0002


real_label = .8
fake_label = 0


def train(gen, disc, dataloader, model_name):

    G_opt = torch.optim.Adam(gen.parameters(), lr=0.00008, betas=(0.5, 0.999))
    D_opt = torch.optim.Adam(disc.parameters(), lr=0.00008, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    gen.train()
    disc.train()


    writer = SummaryWriter(log_dir='runs/' + model_name + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    # writer.add_hparams({'lr': learning_rate, 'bsize': batch_size})
    batch = next(iter(dataloader))
    full_imgs_static, cut_imgs_static, classes_static = batch[0].to(device).float(), batch[1].to(device).float(), batch[2].to(device).float()

    num_iters = 1


    for epoch in range(num_epochs):
        print(' ----- Epoch {} ----- '.format(epoch))
        cur_iter = 0
        for batch in tqdm(dataloader):
            # print('working')
            # if cur_iter >= 1000:
            #     break
            full_imgs, cut_imgs, classes = batch[0].to(device).float(), batch[1].to(device).float(), batch[2].to(device).float()
            # print(classes_onehot)
            labels = torch.ones((len(full_imgs), 1)).to(device)

            # ----- Discriminator Training -----

            # real examples, with label 1
            disc.zero_grad()
            labels.fill_(real_label)
            real_predictions = disc(full_imgs, classes)
            real_Dloss = criterion(real_predictions, labels)
            real_Dloss.backward()
            real_acc = real_accuracy(real_predictions)
            # print('real', real_predictions)

            # fake examples, with label 0
            labels.fill_(fake_label)
            z = torch.rand(size=(len(full_imgs), 100)).to(device)
            fake_imgs = gen(cut_imgs, classes, z)
            fake_predictions = disc(fake_imgs.detach(), classes)
            fake_Dloss = criterion(fake_predictions, labels)
            fake_Dloss.backward()
            fake_acc = fake_accuracy(fake_predictions)
            # print('fake', fake_predictions)

            D_opt.step()
            disc_loss = fake_Dloss + real_Dloss

            # ----- Generator Training -----
            gen.zero_grad()
            labels.fill_(1)
            # z = torch.rand(size=(len(full_imgs), 100)).to(device)
            # fake_imgs = gen(cut_imgs, classes, z)
            Gloss = criterion(disc(fake_imgs, classes), labels)
            Gloss.backward()
            G_opt.step()

            # writer.add_scalar('Loss/Generator', gan_Gloss, num_iters)
            writer.add_scalar('Loss/GeneratorTotal', Gloss, num_iters)
            writer.add_scalar('Loss/Discriminator', disc_loss, num_iters)
            writer.add_scalars('Accuracy', {'Real':real_acc, 'Fake':fake_acc}, num_iters)
            num_iters += 1
            cur_iter += 1

            if num_iters % 250 == 0 or epoch == num_epochs - 1:
                fig = plot_fake_images(gen, full_imgs_static, cut_imgs_static, classes_static, 10)
                writer.add_figure('Generated Images', fig, global_step=num_iters)

            if num_iters % 1000 == 0 or epoch == num_epochs - 1:
                root = 'models/' + model_name + '/'
                if not os.path.exists(root):
                    os.mkdir(root)
                torch.save(gen.state_dict(), root + 'gen_' + str(num_iters // 1000) + '.pt')
                torch.save(disc.state_dict(), root + 'disc_' + str(num_iters // 1000) + '.pt')

        # checkpoint
        # if epoch % 1 == 0 or epoch == num_epochs - 1:
        #     root = 'models/' + model_name + '/'
        #     if not os.path.exists(root):
        #         os.mkdir(root)
        #     torch.save(gen.state_dict(), root + 'gen_' + str(epoch) + '.pt')
        #     torch.save(disc.state_dict(), root + 'disc_' + str(epoch) + '.pt')

        # if epoch % 5 == 0 or epoch == num_epochs - 1:
        #     fig = plot_fake_images(gen, cut_imgs, classes, 10)
        #     writer.add_figure('Generated Images', fig, global_step=epoch)


# https://github.com/Zeleni9/pytorch-wgan/blob/master/models/wgan_gradient_penalty.py
def get_infinite_batches(data_loader):
        while True:
            for i, (images, cutoffs, classes) in enumerate(data_loader):
                yield images.to(device).float(), cutoffs.to(device).float(), classes.to(device).float()

def train_wgan(gen, disc, dataloader, model_name):
    learning_rate = 0.0001
    G_opt = torch.optim.Adam(gen.parameters(), lr=learning_rate * 1, betas=(0, 0.9))
    D_opt = torch.optim.Adam(disc.parameters(), lr=learning_rate * 1, betas=(0, 0.9))
    gen, G_opt = amp.initialize(gen, G_opt, opt_level="O1")
    disc, D_opt = amp.initialize(disc, D_opt, opt_level="O1")

    gen = nn.DataParallel(gen)
    disc = nn.DataParallel(disc)



    gen.train()
    disc.train()
    gp_lambda = 10.
    n_critic_runs = 5

    writer = SummaryWriter(log_dir='runs/' + model_name + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    # writer.add_hparams({'lr': learning_rate, 'bsize': batch_size})
    num_iters = 1

    data = get_infinite_batches(dataloader)
    batch = next(data)
    full_imgs_static, cut_imgs_static, classes_static = batch[0].to(device).float(), batch[1].to(device).float(), batch[2].to(device).float()


    for epoch in range(num_epochs):
        print(' ----- Epoch {} ----- '.format(epoch))

        for g_iter in tqdm(range(500)):

            for p in disc.parameters():
                p.requires_grad = True

            for _ in range(n_critic_runs):
                # ----- Discriminator Training -----
                one = torch.tensor(1, dtype=torch.float)
                mone = one * -1
                one = one.to(device)
                mone = mone.to(device)

                # real examples
                D_opt.zero_grad()
                full_imgs, cut_imgs, classes = next(data)
                # classes = classes.view(-1, 1)
                real_predictions = disc(full_imgs, classes)
                real_predictions = real_predictions.mean()
                # real_predictions.backward(mone)

                with amp.scale_loss(real_predictions, D_opt) as scaled_loss:
                    scaled_loss.backward(mone)

                # fake examples
                z = torch.rand(size=(len(full_imgs), 100)).to(device)
                fake_imgs = gen(cut_imgs, classes, z)
                fake_predictions = disc(fake_imgs.detach(), classes)
                fake_predictions = fake_predictions.mean()
                # fake_predictions.backward(one)

                with amp.scale_loss(fake_predictions, D_opt) as scaled_loss:
                    scaled_loss.backward(one)

                eps = torch.rand(len(full_imgs), 1, 1, 1).to(device)
                eps = eps.expand(full_imgs.shape[0], full_imgs.shape[1], full_imgs.shape[2], full_imgs.shape[3])
                x_hat = eps * full_imgs + (1 - eps) * fake_imgs.detach()
                x_hat = Variable(x_hat, requires_grad=True)
                disc_x_hat = disc(x_hat, classes)

                x_hat_grad = torch.autograd.grad(outputs=disc_x_hat, inputs=x_hat,
                                   grad_outputs=torch.ones(
                                       disc_x_hat.size()).to(device),
                                   create_graph=True, retain_graph=True)[0]

                x_hat_grad = x_hat_grad.view(x_hat_grad.size(0), -1)

                grad_penalty = ((x_hat_grad.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
                # grad_penalty.backward()

                with amp.scale_loss(grad_penalty, D_opt) as scaled_loss:
                    scaled_loss.backward()

                critic_loss = real_predictions - fake_predictions
                D_opt.step()
                # print(disc.module.attention.gamma)

            for p in disc.parameters():
                p.requires_grad = False  # to avoid computation

            # ----- Generator Training -----
            G_opt.zero_grad()
            z = torch.rand(size=(len(full_imgs), 100)).to(device)
            fake_imgs = gen(cut_imgs, classes, z)
            fake_predictions = disc(fake_imgs, classes)
            fake_predictions = fake_predictions.mean()
            # fake_predictions.backward(mone)

            with amp.scale_loss(fake_predictions, G_opt) as scaled_loss:
                scaled_loss.backward(mone)

            Gloss = -fake_predictions

            G_opt.step()

            # gan_Gloss = -fake_predictions


            writer.add_scalar('Loss/Generator', Gloss, num_iters)
            # writer.add_scalar('Loss/GeneratorTotal', new_Gloss, num_iters)
            writer.add_scalar('Loss/Critic', critic_loss, num_iters)
            # print(gen.module.attention2.gamma)
            writer.add_scalar('Extra/Generator_Gamma', gen.module.attention2.gamma.item(), num_iters)
            writer.add_scalar('Extra/Gradient_Penalty', grad_penalty, num_iters)
            num_iters += 1

            if num_iters % 50 == 0 or epoch == num_epochs - 1:
                fig = plot_fake_images(gen, full_imgs_static, cut_imgs_static, classes_static, batch_size)
                writer.add_figure('Generated Images', fig, global_step=num_iters)

        # checkpoint
        if epoch % 1 == 0 or epoch == num_epochs - 1:
            root = 'models/' + model_name + '/'
            if not os.path.exists(root):
                os.mkdir(root)
            torch.save(gen.state_dict(), root + 'gen_' + str(epoch) + '.pt')
            torch.save(disc.state_dict(), root + 'disc_' + str(epoch) + '.pt')


def run():
    torch.multiprocessing.freeze_support()
    gen = SA_Generator(imsize=256, F=64).to(device)
    disc = SA_Discriminator(imsize=256, F=64).to(device)
    # transform = transforms.Compose([transforms.RandomAffine(degrees=10,
    #                                                 translate=(0.1, 0.1)),
    #                            transforms.ToTensor()])
    # transform = transforms.ToTensor()
    # data = MNIST_traindata('data/', transform)
    data = Quickdraw_traindata('data/simplified_tiny/')
    # data = torch.load('data/quickdraw_dataset.pt')
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    train_wgan(gen, disc, dataloader, 'quickdraw_amp')

if __name__ == '__main__':
    run()
