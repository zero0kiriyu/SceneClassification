import pandas as pd
import os

def preprocess():
    """
    listing all the training and testing image's path and its label
    """
    train_path = os.listdir('training')
    class2label = {'Forest': 6, 'bedroom': 0, 'Office': 13,
                   'Highway': 7, 'Coast': 5, 'Insidecity': 8, 'TallBuilding': 12,
                   'industrial': 2, 'Street': 11, 'livingroom': 4,
                   'Suburb': 1, 'Mountain': 9, 'kitchen': 3, 'OpenCountry': 10,
                   'store': 14}

    train_list = pd.DataFrame(columns=['path', 'class', 'label'])

    for f in train_path:  # collect the path and label of train images
        if f == '.DS_Store' or f == '.ipynb_checkpoints':
            continue
        for image in os.listdir('training/'+f):
            if image == '.DS_Store' or image == '.ipynb_checkpoints':
                continue
            train_list = train_list.append(
                {'path': 'training/'+f+'/'+image, 'class': f, 'label': str(class2label[f])}, ignore_index=True)

    train_list.to_csv('train_list.csv', index=None)  # save the output

    test_path = os.listdir('testing')
    test_list = pd.DataFrame(columns=['path', 'predict_label'])

    for image in test_path:  # collect the path of test images
        if image == '.DS_Store' or image == '.ipynb_checkpoints':
            continue
        test_list = test_list.append(
            {'path': 'testing/'+image, 'predict_label': str(-1)}, ignore_index=True)

    test_list.to_csv('test_list.csv', index=None)  # save the output
    return True
