import re
import os
import sys
import pickle
import multiprocessing as mp
from glob import glob
from time import time
from random import Random
import cv2
import numpy as np
import pandas as pd
import sklearn.metrics as skl_metrics
import matplotlib.pyploy as plt
try:
    from tensorflow import keras
except:
    import keras

print('STEP 1: MODEL DEFINITION')

depth = lambda x: x.shape[-1]
if keras.backend.backend() != 'tensorflow':
    depth = lambda x: x.shape.dims[-1]

def cn_block(x, filters, kernel, stride, name='cn', ridge=0.0005):
    ''' convolutional block = conv2d + batchnorm + relu '''
    r = keras.regularizers.l2(ridge)
    x = keras.layers.Conv2D(filters, kernel, strides=stride, padding='same',
            use_bias=False, kernel_regularizer=r, name=f'{name}_conv')(x)
    x = keras.layers.BatchNormalization(name=f'{name}_bn')(x)
    return keras.layers.Activation('relu')(x)

def fc_block(x, out, name='fc', ridge=0.0005):
    ''' fully-connected (dense) block + dropout '''
    r = keras.regularizers.l2(ridge)
    x = keras.layers.Dense(out, use_bias=False, kernel_regularizer=r, name=name)(x)
    return keras.layers.Dropout(0.5)(x)

def se_block(i, out, name='se', reduction=2):
    ''' squeeze-excite attention block '''
    x = keras.layers.GlobalAvgPool2D()(i)
    x = fc_block(x, out//reduction, name=f'{name}_fc1')
    x = keras.layers.Activation('relu')(x)
    x = fc_block(x, out, name=f'{name}_fc2')
    x = keras.layers.Activation('sigmoid')(x)
    x = keras.layers.Reshape((1,1,out))(x)
    return keras.layers.multiply([i,x])

def id_block(i, out, num, cardinality1=3, cardinality2=8, split_out=8):
    ''' aggregated residual + se block '''
    ch = depth(i)
    if ch == out:
        s, p = 1, i
    else:
        assert(out == 2*ch)
        s = 2
        p = keras.layers.AvgPool2D()(i)
        p = keras.layers.concatenate([p,p])
    xr = []
    for j in range(cardinality1):
        xs = []
        for k in range(cardinality2):
            x = cn_block(i, split_out, (1,1), 1, f'res_split{num}{j:x}{k:x}1')
            x = cn_block(x, split_out, (3,3), s, f'res_split{num}{j:x}{k:x}2')
            xs.append(x)
        x = keras.layers.concatenate(xs)
        x = cn_block(x, out, (1,1), 1, name=f'res_tr{num}{j:x}')
        x = se_block(x, out, f'res_se{num}{j:x}')
        x = keras.layers.add([p,x])
        x = keras.layers.Activation('relu')(x)
        xr.append(x)
    x = keras.layers.concatenate(xr)
    x = cn_block(x, out, (1,1), 1, name=f'res_join{num}')
    return x

def DermaDL(input_shape=(96,96,3), outputs=5, activation='softmax', d_out=128, **opts):
    ''' our network with multi-class output '''
    i = keras.Input(input_shape)
    x = cn_block(i, d_out//8, (3,3), 1, 'init')
    x = id_block(x, d_out//8, 0, **opts)
    x = id_block(x, d_out//4, 1, **opts)
    x = id_block(x, d_out//2, 2, **opts)
    x = id_block(x, d_out//1, 3, **opts)
    x = keras.layers.GlobalAvgPool2D(name='last_pool')(x)
    x = fc_block(x, d_out, name='last_fc')
    x = keras.layers.Dense(outputs, activation=activation, name='logits')(x)
    return keras.Model(i,x)

def get_df(path, balance=True, shuffle=True, samples=None, seed=999991):
    ''' dataset loading helper '''
    rng = Random(seed)
    data = []
    for classdir in glob(os.path.join(path,'*')):
        classname = classdir.split(os.path.sep)[-1]
        files = [] 
        for filename in glob(os.path.join(classdir,'*')):
            files.append([filename, classname])
        if shuffle:
            rng.shuffle(files)
        if samples:
            files = files[:samples]
        data.append(files)
    df = []
    if balance:
        n = max([len(x) for x in data])
        for l in [rng.choices(x,k=n) for x in data]:
            df.extend(l)
    else:
        for l in data:
            df.extend(l)
    if shuffle:
        rng.shuffle(df)
    return pd.DataFrame(df, columns=['filename','class'])

print('STEP 2: DATA PREPARATION - METADATA NORMALIZATION')

if os.path.exists('labels.csv'):
    print('\tMetadata is already prepared, skipping...') 
else:
    tr = {
        'actinic keratosis':'keratosis',
        'angiofibroma or fibrous papule':'other',
        'angioma':'other',
        'atypical melanocytic proliferation':'nevus',
        'basal cell carcinoma':'carcinoma',
        'blue nevus':'nevus',
        'clark nevus':'nevus',
        'combined nevus':'nevus',
        'congenital nevus':'nevus',
        'dermal nevus':'nevus',
        'dermatofibroma':'other',
        'dysplastic nevus':'nevus',
        'intradermal nevus':'nevus',
        'lentigo':'other',
        'lentigo NOS':'other',
        'lentigo maligna':'melanoma',
        'lentigo simplex':'other',
        'lichenoid keratosis':'keratosis',
        'melanoma':'melanoma',
        'melanoma metastasis':'melanoma',
        'melanosis':'other',
        'miscellaneous':'other',
        'nevus':'nevus',
        'nodular melanoma':'melanoma',
        'other':'other',
        'pigmented benign keratosis':'keratosis',
        'recurrent nevus':'nevus',
        'reed or spitz nevus':'nevus',
        'scar':'other',
        'seborrheic keratosis':'keratosis',
        'solar lentigo':'other',
        'squamous cell carcinoma':'carcinoma',
        'vascular lesion':'other',
        None:'other',
    }

    data = []

    df = pd.read_csv('./datasets/release_v0/meta/meta.csv')
    for _,t in df.iterrows():
        diag = tr[t.diagnosis.split(' (')[0]]
        path1 = '7pt/'+os.path.basename(t.clinic)
        path2 = '7pt/'+os.path.basename(t.derm)
        data.append(['7pt', path1, diag, 'clinic'])
        data.append(['7pt', path2, diag, 'derm'])

    df = pd.read_excel('./datasets/PH2Dataset/PH2_dataset.xlsx', skiprows=12)
    df = df.fillna('other')
    for _,t in df.iterrows():
        path = 'PH2/'+t['Image Name']+'.jpg'
        diag = t['Histological Diagnosis']
        diag = tr[diag.lower()]
        data.append(['PH2', path, diag, 'derm'])

    df = pd.read_json('../datasets/ISIC/metadata.json')
    for _,t in df.iterrows():
        if len(t['dataset']['name']) < 9:
            path = 'ISIC/'+t['dataset']['name']+'/'+t['name']+'.jpg'
            diag = tr[t['meta']['clinical'].get('diagnosis')]
            data.append(['ISIC', path, diag, 'derm'])

    df = pd.DataFrame(data, columns='set img label type'.split())
    df.to_csv('labels.csv')

print('STEP 3: DATA PREPARATION - EQUALIZE, CROP AND RESIZE IMAGES')

def crop_bb(im, bw):
    h,w = bw.shape
    fi = np.zeros((h+4,w+4), dtype='uint8')
    fi[2:-2,2:-2] = bw.astype('uint8')
    fi = cv2.medianBlur(fi, 5)[2:-2,2:-2]
    _, u = cv2.threshold(fi,  25, 1, cv2.THRESH_BINARY)
    _, v = cv2.threshold(fi, 220, 1, cv2.THRESH_BINARY_INV)
    t = cv2.morphologyEx(u*v, cv2.MORPH_OPEN, np.ones((3,3)))
    i1,i0,j1,j0 = cv2.boundingRect(t)
    im = im[i0:i0+j0, i1:i1+j1]
    bw = bw[i0:i0+j0, i1:i1+j1]
    im = cv2.resize(im, (500,500), interpolation=cv2.INTER_CUBIC)
    bw = cv2.resize(bw, (500,500), interpolation=cv2.INTER_CUBIC)
    return im, bw

def imsave(tag, path, img):
    out = os.path.join(tag, path)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, img)

def task(path):
    base, ext = os.path.splitext(path)
    out = base.replace('datasets','clashed2') + '.jpg'
    if os.path.exists(os.path.join('prep3', out)):
        return
    try:
        eq = cv2.createCLAHE(2.0, (8,8))
        im = cv2.imread(path)
        bw = cv2.imread(path,0)
        im, bw = crop_bb(im,bw)

        e1 = eq.apply(bw)
        imsave('prep1', out, e1)

        e2 = np.transpose([eq.apply(x) for x in im.T])
        imsave('prep2', out, e2)

        e3 = cv2.cvtColor(im, cv2.COLOR_BGR2HLS).T
        e3 = np.transpose([e3[0], eq.apply(e3[1]), e3[2]])
        e3 = cv2.cvtColor(e3, cv2.COLOR_HLS2BGR)
        imsave('prep3', out, e3)
    except:
        print(path, 'failed')

# OBS: download lesion images into ./datasets/*
#      organize ISIC by source datasets, e.g. ./datasets/ISIC/HAM10000/ISIC_0021440.jpg, 
#      7pt/PH2: only images, e.g. .datasets/7pt/Aal001.jpg and .datasets/PH2/IMD002.jpg

files = sorted(glob('./datasets/ISIC/*/*.jpg') + glob('../datasets/7pt/*.jpg') + glob('PH2/*.bmp'))
if os.path.exists('./clashed2/prep3/PH2'):
    print('\tImages already preprocessed, skipping...')
else:
    print(f'\t{len(files)} files to go, starting {os.cpu_count()} workers.')
    t0 = time()
    with mp.Pool() as pool:
        pool.map(task, files)
    pool.join()
    print(f'\tFinished in {time()-t0:.3f}s.')

print('STEP 4: DATA AUGMENTATION')

if os.path.exists('./images2/prep1/PH2'):
    print('\tImages already augmented, skipping...')
else:
    for prep in [3,2,1]:
        df = get_df(f'clashed2/prep{prep}')
        dgen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            zoom_range=[0.7, 1.0],
            shear_range=0.05,
            rotation_range=15.0,
            width_shift_range=0.1,
            height_shift_range=0.1,
            fill_mode='reflect',
            vertical_flip=True,
            horizontal_flip=True,
            brightness_range=[0.8, 1.0],
            channel_shift_range=0.1,
        ).flow_from_dataframe(
            df,
            target_size=(144,144),
            interpolation='bicubic',
            batch_size=1,
            shuffle=False,
            seed=999991,
            class_mode='categorical',
        )
        for c in dgen.class_indices:
            os.makedirs(f'images2/prep{prep}/{c}', exist_ok=True)
        for i,(f,t) in enumerate(zip(dgen.filenames, dgen)):
            g = f.replace('clashed2','images2').replace('.jpg',f'_{i}.jpg')
            keras.preprocessing.image.save_img(g, t[0][0])

print('STEP 5: TRAINING')

loss, f = 'categorical_crossentropy', 'softmax'
src_size=144
net_size=(96,96)
prep=3
bsize=32
epochs=100
out_dir = f'runs/prep{prep}_{loss.split("_")[1]}'
os.makedirs(out_dir, exist_ok=1)

model = DermaDL(net_size+(3,), activation=f, d_out=232, cardinality1=9, cardinality2=9, split_out=8)

df_train = get_df(f'images2/prep{prep}')
df_train = df_train.iloc[:int(0.9*len(df_train))] # first 90% is train+validation data

dgen_train = keras.preprocessing.image.ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    validation_split=1/9, # validation is 1/9 of 90%, i.e. 10% of all data
)
data_train = dgen_train.flow_from_dataframe(
    df_train,
    target_size=net_size,
    interpolation='bicubic',
    batch_size=bsize,
    shuffle=False,
    seed=999991,
    class_mode='categorical',
    subset='training',
)
data_valid = dgen_train.flow_from_dataframe(
    df_train,
    target_size=net_size,
    interpolation='bicubic',
    batch_size=bsize,
    shuffle=False,
    seed=999991,
    class_mode='categorical',
    subset='validation',
)

optim = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, decay=0.00005, nesterov=True)
model.compile(optim, loss, metrics=['categorical_accuracy'])
print('Model size:',model.count_params())
with open(f'{out_dir}/model.json','w') as f:
    f.write(model.to_json())
h = model.fit(data_train, validation_data=data_valid, epochs=epochs, batch_size=bsize,
        callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                   keras.callbacks.ModelCheckpoint(f'{out_dir}/weights.h5',
                       save_best_only=True, save_weights_only=True)])
with open(f'{out_dir}/history.pkl','wb') as f:
    pickle.dump(h.history, f)
model.save_weights(f'{out_dir}/weights.h5')
print('Saved as:',out_dir)

print('STEP 6: EVALUATION')

src_size = 144
net_size = (96,96)
basedir = out_dir

df_test = get_df(f'images2/prep{prep}/all')
df_test = df_test.iloc[int(0.9*len(df_test)):] # last 10% is test data

data_test = keras.preprocessing.image.ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
).flow_from_dataframe(
    df_test,
    target_size=net_size,
    interpolation='bicubic',
    batch_size=1,
    shuffle=False,
    seed=123456,
    class_mode='categorical',
)

with open(f'{basedir}/model.json','r') as f:
    model = keras.models.model_from_json(f.read())
model.load_weights(f'{basedir}/weights.h5')

def score(y,p):
    cm = skl_metrics.confusion_matrix(y, p.round())
    tpr = cm[1,1]/cm[1].sum() # tpr/sensitivity
    tnr = cm[0,0]/cm[0].sum() # tnr/specificity
    return cm, [
        skl_metrics.roc_auc_score(y, p),
        skl_metrics.accuracy_score(y, p.round()),
        skl_metrics.precision_score(y, p.round()), #also ppv
        #skl_metrics.recall_score(y, p.round()),   #also tpr
        tpr, tnr, tpr+tnr-1
    ]

def printscores(metrics, title=''):
    print(f'--{title}\nClass\tAUC\tACC\tPPV\tTPR\tTNR\tS50+')
    for k,i in data_test.class_indices.items():
        print(f'{k[:6]}', end='')
        for m in metrics[i]:
            print(f'\t{m:.4f}', end='')
        print('')
    print('Mean', end='')
    for m in np.mean([m for m in metrics if len(m)>0], 0):
        print(f'\t{m:.4f}', end='')
    print('')

p = model.predict(data_test)
y = keras.utils.to_categorical(data_test.classes)
cms, metrics = [], []
for yi,pi in zip(y.T,p.T):
    s = score(yi,pi)
    cms.append(s[0])
    metrics.append(s[1])

dsets = { 'isic':[], '7pt':[], 'ph2':[] }
tr = {'IS':'isic', 'IM':'ph2', }
for yj,pj,f in zip(y,p,data_test.filenames):
    d = tr.get(os.path.basename(f)[:2], '7pt')
    dsets[d].append([yj, pj])

for d in list(dsets):
    a = np.array(dsets[d])
    yj, pj, m = a[:,0], a[:,1], []
    for yi,pi in zip(yj.T,pj.T):
        if np.unique(yi).size > 1:
            m.append(score(yi,pi)[1])
        else:
            m.append([])
    printscores(m, d)

printscores(metrics, 'ALL')

conf = skl_metrics.confusion_matrix(np.argmax(y,1), np.argmax(p,1))
print(conf) # rows are true conditions, columns are predictions
