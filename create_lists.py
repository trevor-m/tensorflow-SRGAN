import glob
import os
import pickle
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('train', type=str, help='Training directory')
parser.add_argument('val', type=str, help='Validation directory')
args = parser.parse_args()

train_base = args.train #input('Enter training directory(will be recursively searched):')
val_base = args.val #input('Enter validation directory(will be recursively searched):')

train_files = glob.glob(os.path.join(train_base, '**', '*.*'), recursive=True)
#types = set([x.split('.')[-1] for x in train_files])
#print(types)
#train_files.extend(glob.glob(os.path.join(train_base, '**', '*.png'), recursive=True))
#train_files.extend(glob.glob(os.path.join(train_base, '**', '*.bmp'), recursive=True))
#train_files.extend(glob.glob(os.path.join(train_base, '**', '*.jpeg'), recursive=True))
pickle.dump(train_files, open('train.pickle', 'wb'))

val_files = glob.glob(os.path.join(val_base, '**', '*_HR.png'), recursive=True)
pickle.dump(val_files, open('val.pickle', 'wb'))

eval_indices = np.random.randint(len(train_files), size=len(val_files))
pickle.dump(eval_indices, open('eval_indexes.pickle', 'wb'))
print(len(train_files), len(val_files), len(eval_indices))