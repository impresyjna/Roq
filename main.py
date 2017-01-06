import graphlab as gl
import numpy as np

X_train = gl.SFrame.read_csv("data/X_train.csv",header=False)
y_train = gl.SFrame.read_csv("data/y_train.csv",header=False)

X_valid = gl.SFrame.read_csv("data/X_valid.csv",header=False)
y_valid = gl.SFrame.read_csv("data/y_valid.csv",header=False)

X_train['classification'] = y_train['X1']
X_valid['classification'] = y_valid['X1']

X_test = gl.SFrame.read_csv("data/X_test.csv",header=False)

model = gl.classifier.create(X_train, target='classification')

# Generate predictions (class/probabilities etc.), contained in an SFrame.
predictions = model.classify(X_valid)

# # Evaluate the model, with the results stored in a dictionary
# results = model.evaluate(X_valid)

roq_test_predictions = model.classify(X_test)
print roq_test_predictions

print roq_test_predictions['class']
outstring = zip(list(range(1,len(roq_test_predictions['class'])+2)), roq_test_predictions['class'])
f = open('answer.csv', 'w')
f.write('id,predicted\n')
for line in outstring:
    f.write(",".join(str(x) for x in line) + "\n")
f.close()

# answer_file = open('answer.csv', 'w')
# answer_file.write('id,predicted\n')
# for i in range(len(roq_test_predictions['class'])):
#     line = str(i+1)+','+str(roq_test_predictions['class'][i])+'\n'
#     answer_file.write(line)
# print len(roq_test_predictions['class'])
# answer_file.close()
# answear[0]=list(range(1,len(roq_test_predictions['class'])))
# answear[1] = roq_test_predictions['class']
# print answear.T


