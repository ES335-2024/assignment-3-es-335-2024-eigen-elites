We can observe that Random Forest gives us the highest F1 score and accuracy followed by MLP and logistic regression. 5-8 and 4-9 are some of the commonly confused digits. 

We trained our model for 10 epochs and another model was left untrained. Since the untrained model's weights and biases are initialized randomly, the layers will not capture any meaningful patterns or structures in the data. So, there is no clear separation of classes in our plot. If the trained model has learned meaningful representations, we would expect to see clearer and more separated clusters compared to the untrained model.

Now we predict on the Fashion MNIST dataset using our trained model (trained on MNIST). Our model confidently predicts the label 2 and 3 for most training datapoints. There is clearly an issue here. We cannot expect that our model will always encounter the kind of data it is trained on. But our model which is trained for numbers should not be predicting for clothing articles however we are still making confident predictions. This is out-of-distribution (OOD) detection.

While there aren't clear clusters embeddings for the fashion mnist dataset, there is still a greater division as compared to the untrained model for mnist. So trained MNIST > trained fashion MNIST > untrained MNIST is the order for clear clusters.
