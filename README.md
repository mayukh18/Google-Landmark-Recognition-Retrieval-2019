# Google Landmark Recognition & Retrieval Challenges 2019

This solution ranked 31st in the [Landmark Recognition Challenge](https://www.kaggle.com/c/landmark-recognition-2019/overview) and 38th in the [Landmark Retrieval Challenge](https://www.kaggle.com/c/landmark-retrieval-2019/overview) and has code suitable for running in the Kaggle Kernels.

| Challenge | Public LB GAP | Private LB GAP |
| --- | --- | --- |
| Recognition | 0.09273 | 0.10926 |
| Retrieval | 0.09373 | 0.11383 |

The code in Keras has been divided into separate files for better organization. The `recognition.py` file strings everything together to train the model and create the submission for the recognition challenge. The `retrieval.py` file loads the model trained for the recognition problem and creates the submission.

The pretrained model can be downlaoded [here](https://www.kaggle.com/mayukh18/resnet50-0092#resnet50.model).




### Dataset

The dataset is available at [CVDF](https://github.com/cvdfoundation/google-landmark). The train set comprises of around 5 million images (500GB) for train set, around 700K images for index set (retrieval problem) and 110K test images in test set.

### Approach for Kaggle Kernels

The solution was run completely on Kaggle Kernels. Since the dataset size is around 500 GB, the training was done on a download-train-discard methodology. In each mini-epoch, the data generators downloaded one tar file from the 500 tar files; extracted it; created batches from the valid images and at the end of the epoch deleted all the files. The generator used can be found in `generators.py`.

### Model Architecture

- I used a ResNet50 with weights pretrained on imagenet. Chopped off of the final layer. Added a GlobalMaxPooling layer and a subsequent softmax layer at the end for the output.
- Used landmarks which have atleast 20 images.(~55k landmarks)
- Resized every image to 192x192 pixels.
- Used Adam optimizer with a rate of 0.0002 at the staring and reduced it after each ~150 mini-epochs.
- Trained the model for one full epoch(500 mini-epochs) after which it started overfitting. Could have tuned the hyperparameters better for the subsequent epochs but there was not much time left for experimentation.
- Did not use any external dataset for non landmark images.
- TTA somehow gave me worse results.

Special mention to Dawnbreaker's [solution](https://www.kaggle.com/c/landmark-recognition-challenge/discussion/57152) from 2018 version of the challenge which helped me improve my results.

I initially started off with a ResNet101 but it failed to match the level of performance of the ResNet50. Taking this from a comment of Dawnbreaker in the above solution thread:
> "Actually I tried ResNet101 and got a higher val accuracy but a worse LB performance (LB:0.101). Eventually I think maybe a accurate prediction for the landmark images is not as important as an efficient rejection for the non-landmark images."

Perhaps this makes sense. Add to the fact that here we were predicting on ~55k classes compared to last year's 15k, rejection seems even more important.

### Retrieval Solution

- Used the same model trained in recognition. Chopped off the softmax layer and used the pooling layer outputs as features.
- Used [faiss](https://github.com/facebookresearch/faiss) as it is superfast for similarity search.
- Used it to find top 100 similar images to each test set image. To fit it into kernels, I had to do and improper way of getting an approximate top 100. Broke down the index set to 6 kernels, found 6 sets of top 100 matches from each of the 6 parts of the index set, then selected the top 100 among the 600 matches based on distances.
- A hack I applied with my very last available submission boosted my LB score by about 50%. What I did was to use the original complete model to predict landmarks for both index and test set. Now for each of the test image if the predicted landmark matches with that of an index image, I divide the corresponding distance by 1.5.