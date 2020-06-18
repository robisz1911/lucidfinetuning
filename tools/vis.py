import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn import manifold, decomposition
from skimage.transform import rescale, resize
from matplotlib.offsetbox import AnnotationBbox, OffsetImage


activations_file, prefix =  sys.argv[1:]

a = np.load(activations_file)['arr_0']
a = a.T

print("TRUNCATING") ; a = a[-128:, :]


def diag_stats():
    print(np.mean(a.diagonal()))
    for i in range(5):
        s = a.copy()
        np.random.shuffle(s)
        print(np.mean(s.diagonal()))
    print()
    print(np.mean(a))
    print("=====")


diag_stats()

print("NORMALIZING NEURONS")
a -= a.mean(axis=1, keepdims=True)
a /= a.std(axis=1, keepdims=True)

diag_stats()

# plt.imshow(a) ; plt.show()


def load_images(prefix, cnt):
    nonsquare = False
    imgs = []
    for i in range(cnt):
        fname = prefix + ("%03d" % i) + ".png"
        img = plt.imread(fname, format='png')
        if img.shape[0] != img.shape[1]:
            s = 224
            img = img[:s, :s, :]
            if not nonsquare: # printing only once
                print("Keeping only top left %d x %d of each image." % (s, s))
            nonsquare = True
        img = resize(img, (224, 224), anti_aliasing=True)
        img = (img * 255).astype(np.uint8)
        imgs.append(img)
    return np.array(imgs)

imgs = None
imgs = load_images(prefix, cnt=a.shape[0]) ; print(imgs.shape)
# np.save("pure-lucid-224.npy", imgs)

mags = np.linalg.norm(a, axis=1)
# plt.hist(mags, bins=50)
# plt.show()

a /= mags[:, np.newaxis]

projector = manifold.TSNE(2, metric='cosine', perplexity=10)
# projector = decomposition.PCA(2)

r = projector.fit_transform(a)
print(r.shape)

fig = plt.figure(figsize=(40, 40))

ax = plt.subplot(111)

ax.scatter(r[:, 0], r[:, 1], c=mags)

if imgs is not None:
    for x, y, img in zip(r[:, 0], r[:, 1], imgs):
        imagebox = OffsetImage(img, zoom=.5)
        xy = [x, y]               # coordinates to position this image

        ab = AnnotationBbox(imagebox, xy,
            xybox=(10., -10.),
            xycoords='data',
            boxcoords="offset points",
            pad=0.0)
        ax.add_artist(ab)

plt.savefig("tsne.png")
