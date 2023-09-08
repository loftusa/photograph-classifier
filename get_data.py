#%%
!pip install duckduckgo_search
from duckduckgo_search import ddg_images
from fastcore.all import *
from fastai.vision.all import *
from fastdownload import download_url

#%%
def search_images(term, max_images=30):
    print(f"Searching for '{term}'")
    return L(ddg_images(term, max_results=max_images)).itemgot('image')
    

# for term in ["sun", "city", "forest", "animal", "cloud", "mountain", "water", "flower", "tree", "building", "car", "person"]:
#     download_images(max_pics=1000, dest='photos', urls=search_images(f'{term} photograph'))

#%%
photos = get_image_files('data/photos')
failed = verify_images(photos)
failed.map(Path.unlink)
photos = get_image_files('data/photos')

photo_imgs = photos.map(PILImage.create)
#%%
sample = random.sample(photo_imgs, 9)
show_images(sample, nrows=3, ncols=3)
# %%

# for term in ["sun", "city", "forest", "animal", "cloud", "mountain", "water", "flower", "tree", "building", "car", "person"]:
#     download_images(max_pics=1000, dest='junk', urls=search_images(f'{term} photograph advertisement'))
#%%

junk = get_image_files('data/junk')
failed = verify_images(junk)
failed.map(Path.unlink)
junk = get_image_files('data/junk')

# %%
junk_imgs = junk.map(PILImage.create)
sample = random.sample(junk_imgs, 9)
show_images(sample, nrows=3, ncols=3)
#%%
#%%
import shutil
# move 'junk' and 'photos' into a new directory called 'data'
# Path('data').mkdir(exist_ok=True)

# shutil.move('junk', 'data')
# shutil.move('photos', 'data')

#%%

db = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(192, method='squish'),   
)
dls = db.dataloaders('data')

#%%
dls.show_batch()

#%%

# learner = vision_learner(dls, resnet18, metrics=error_rate)
learner = load_learner("junk_classifier.pkl").to('cuda')
#%%
learner.dls = dls
# %%
learner.fine_tune(epochs=1)
# %%
# learner.export('junk_classifier.pkl')
#%%
# learner.model.to('cuda')

# %%
Path('.').ls()
# %%
# db = db.new(item_tfms=RandomResizedCrop(128, min_scale=0.3), batch_tfms=aug_transforms(mult=2))
# dls = db.dataloaders('data')
# dls.show_batch()
#%%
# %%
# %%
interp = ClassificationInterpretation.from_learner(learner)
# interp2 = ClassificationInterpretation.from_learner(learner2)
#%%
interp.plot_confusion_matrix()
# %%
interp.plot_top_losses(10)
# %%
from fastai.vision.widgets import ImageClassifierCleaner
# %%
cleaner = ImageClassifierCleaner(learner)
cleaner
# %%
cleaner