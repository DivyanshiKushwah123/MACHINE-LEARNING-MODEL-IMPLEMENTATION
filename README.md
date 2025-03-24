# MACHINE-LEARNING-MODEL-IMPLEMENTATION

company : CODTECH IT SOLUTION

NAME : DIVYANSHI KUSHWAH

*INTERN ID * : CT08WQT

DOMAIL : PYTHON PROGRAMMING

DURATION :4 WEEKS

MENTOR :NEELA SANTHOSH

DESCRIPTION
This Python script performs spam email detection using Logistic Regression, a popular machine learning model for binary classification tasks. It begins by importing essential libraries such as `pandas` for data manipulation, `numpy` for numerical operations, `sklearn` for machine learning tools, and `matplotlib` and `seaborn` for data visualization. The script reads a dataset called 'spam.csv', which contains two columns: one representing the label of the email (either "spam" or "ham") and the other containing the actual message. The data is cleaned by renaming the columns for clarity and checking for missing values, ensuring the dataset is ready for analysis. To convert the email messages into numerical features that a machine learning model can understand, the script utilizes `CountVectorizer` from `sklearn`, which converts the text into a sparse matrix of token counts while removing common English stopwords. The target variable, or label, is then mapped to numerical values, where 'ham' is assigned 0 and 'spam' is assigned 1. The data is split into training and testing sets, with 80% of the data used for training the model and the remaining 20% used for testing. This division helps to evaluate the performance of the model on unseen data. The logistic regression model is initialized and trained using the training data, and then predictions are made on the test set. The script evaluates the model's performance using accuracy, confusion matrix, and classification report. The accuracy score is printed, showing the percentage of correctly classified messages, while the confusion matrix, visualized using a heatmap, provides a detailed comparison of predicted versus actual labels (ham and spam). Additionally, the classification report displays important metrics such as precision, recall, and F1-score, giving further insight into the model's performance for both classes. To test the model with new, unseen messages, two sample messages are provided: one promoting a lottery and the other asking about a meeting. These new messages are transformed into the same format as the training data using the same `CountVectorizer` and then passed through the trained model for predictions. The script outputs whether each new message is predicted as "Spam" or "Ham." This demonstrates the practical application of logistic regression for text classification tasks, specifically for spam email detection. The model is able to automatically classify emails based on the patterns learned from the training data, providing a robust method for filtering unwanted messages. However, the model's performance is dependent on the quality of the dataset and the features used for training, and it could be further enhanced by exploring more advanced natural language processing techniques or by using a more complex model like support vector machines or deep learning for better accuracy and generalization.
