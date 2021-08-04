from tkinter import *
import numpy as np
from adain_model import StyleTransferModel
from torch.utils.data import DataLoader
from data import *
from PIL import Image, ImageTk
from torchvision.transforms import ToTensor

w = Tk()

photo_list = []
nr = 1
nc = 3

m = StyleTransferModel()
state_dict = torch.load("./demo/1625494627.6914184/44.pt", map_location=torch.device('cpu'))['model_state_dict']
m.load_state_dict(state_dict)
transform = get_transforms(False, 500)

COCO_PATH = "datasets/train2017"
COCO_LABELS_PATH = "datasets/annotations/captions_train2017.json"
WIKIART_PATH = "datasets/wikiart"

dataset = StyleTransferDataset(COCO_PATH, COCO_LABELS_PATH, WIKIART_PATH, length = 50,
                               transform = transform, rng_seed = 3033157)
dataloader = DataLoader(dataset, batch_size = 1)
with torch.no_grad():
    D = [(c, s) for c, s in dataloader]

print(D[0][0].max())
print(D[0][0].min())

tt = ToTensor()
with Image.open("./content.jpg") as im1:
    with Image.open("style.jpg") as im2:
        im1 = im1.resize((512, 512))
        im2 = im2.resize((512, 512))
        c = tt(im1).unsqueeze(0)
        s = tt(im2).unsqueeze(0)
        print(c.max())
        print(c.min())
        print(c.shape)
        print(s.shape)
        out = m(c, s)[-1].detach()
        print(out.shape)
        out = out.squeeze(0).permute(1, 2, 0).numpy()
        print(out.shape)
        out = (out * 255).astype(np.uint8)
        image = Image.fromarray(out)
        image.save("out.png", "PNG")
aasda

curr = 0
imgs = D[curr]
bs = []
for i in range(nr*nc):
    #c = imgs[i]
    #a = ImageTk.PhotoImage(Image.fromarray(c))
    #photo_list.append(a)
    b = Label(w, image = None)
    b.grid(row=i//nc, column=i%nc)
    bs.append(b)


def re(*args, **kwargs):
    global curr
    curr += 1
    c = D[curr][0]
    c = (c[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    a = ImageTk.PhotoImage(Image.fromarray(c))
    photo_list.append(a)
    bs[0].configure(image = a)
    bs[0].image = a

    s = D[curr][1]
    s = (s[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    a = ImageTk.PhotoImage(Image.fromarray(s))
    photo_list.append(a)
    bs[1].configure(image = a)
    bs[1].image = a

    with torch.no_grad():
        out = m(D[curr][0], D[curr][1])[-1]
    out = (out[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    a = ImageTk.PhotoImage(Image.fromarray(out))
    photo_list.append(a)
    bs[2].configure(image = a)
    bs[2].image = a


a = Button(w, text = "ASDSADASD", height = 2, width = 30, command = re)
a.grid(row = nr + 1, column = nc // 2)

w.mainloop()
