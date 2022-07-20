import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter.messagebox import showinfo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import sklearn
import os


FILE = None
df = None
X = None
y = None
col = None
pcamodel = None
pca = None


def openFile():
    global File
    global df
    filetypes = (
        ('data file', '*.data'),
        ('csv file', '*.csv'),
        ('All files', '*.*')
    )

    filename = filedialog.askopenfilename(
        title='Open a file',
        initialdir=os.path.curdir + '/data',
        filetypes=filetypes)

    tk.messagebox.showinfo(
        title='Selected File',
        message=filename
    )
    FILE = filename
    df = pd.read_csv(filename)
    print(df.info)


    global X
    global y
    global col
    X = df.drop(["status", "name"], axis=1)
    y = df["status"]
    col = X.columns



def plotCorr():
    plt.figure(
                num="Correlation Matrix",
                figsize=(20,16))

    ax = sns.heatmap(
                data = df.corr(),
                annot= True,
                cmap='YlGnBu'
    )
    plt.show()

def plotDistPlot():
    fig, axes = plt.subplots(6, 4, num="Distribution Plot",figsize=(40, 40))
    for i in range(0, 24):
        if i + 1 < len(df.columns):
            sns.distplot(ax=axes[i//4][i%4], a=df.iloc[:, i + 1])

    plt.subplots_adjust(hspace=0.7, wspace=0.3)
    plt.show()

def plotDistPlot2():
    fig, axes = plt.subplots(6, 4, num="Normalized Dist Plot", figsize=(40, 24))
    for i in range(0, 24):
        if i + 1 < len(X.columns):
            sns.distplot(ax=axes[i//4][i%4], a=X.iloc[:, i])
    plt.subplots_adjust(hspace=0.7, wspace=0.3)
    plt.show()

def plotBoxPlot():
    fig, axes = plt.subplots(6, 4,num="Box Plot", figsize=(20, 40))
    for i in range(0, 24):
        if i + 1 < len(df.columns):
            sns.boxplot(
                    ax=axes[i//4][i%4], 
                    x=df.columns[i + 1],
                    bootstrap=None,
                    data=df)
    plt.subplots_adjust(
                        hspace=0.7, 
                        wspace=0.5)
    plt.show()

def utility():
    global X
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = pd.DataFrame(
                data=X,
                columns=col
    )

def applyPCA():
    global pcamodel
    global pca
    pcamodel = PCA(
              n_components=0.90,
              random_state=42
    )
    pca = pcamodel.fit_transform(X)
    print(pca.shape)
    print(pcamodel.components_.shape)
    print(pcamodel.explained_variance_ratio_)

def plotPCAVarPlot():
    plt.bar(range(1,len(pcamodel.explained_variance_ )+1),pcamodel.explained_variance_ratio_ )
    plt.ylabel('Explained variance')
    plt.xlabel('Components')
    plt.plot(range(1,len(pcamodel.explained_variance_ratio_ )+1),
            np.cumsum(pcamodel.explained_variance_ratio_),
            c='red',
            label="Cumulative Explained Variance")
    plt.legend(loc='upper left')
    plt.show()

def logisticRegression():
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
                                                                            pca,
                                                                            y,
                                                                            test_size=0.2,
                                                                             random_state=42  
    )
    logreg = logisticRegression()
    gridLogReg = {
                'C': np.logspace(-3,3,7),
                'solver': ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                'fit_intercept': [True, False],
                'class_weight': ['balanced', None]
    }
    logregCV = GridSearchCV(logreg, gridLogReg, cv=10)

def myplot(score,coeff,pc,labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley,s=5)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'green', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
 
def plotPCABiPlot():
    plt.figure(num="PCA BiPlot", figsize=(40,40))
    myplot(pca[:,0:2],np.transpose(pcamodel.components_[0:2, :]),None,col)
    plt.show()

plt.show()
window = tk.Tk()
root = window
openFrame = tk.LabelFrame(root, text="Open")
openFrame.pack(side=tk.TOP)
openFileButton = tk.Button(openFrame, text = "Open the data file", fg = "black", command=openFile)  
openFileButton.pack(side = tk.TOP, fill="x", padx=20)

visualFrame = tk.LabelFrame(root, text = "Data Visualization")
visualFrame.pack(side = tk.TOP)
corrButton = tk.Button(visualFrame, text = "Correlation Matrix", command = plotCorr)
corrButton.pack(side=tk.TOP, fill="x", padx=20)

distButton = tk.Button(visualFrame, text = "Distribution Plot", command = plotDistPlot)
distButton.pack(side=tk.TOP, fill="x", padx=20)

boxPlotButton = tk.Button(visualFrame, text = "Box Plot", command = plotBoxPlot)
boxPlotButton.pack(side=tk.TOP, fill="x", padx=20)

preproFrame = tk.LabelFrame(
                root,
                text = "Preprocessing"
)
preproFrame.pack(side=tk.TOP)
standardButton = tk.Button(preproFrame, text="Standard Scaler", command=utility)
standardButton.pack(side=tk.TOP, fill="x", padx=20)
dist2Button = tk.Button(preproFrame, text = "Dist Plot(Standardised)", command = plotDistPlot2)
dist2Button.pack(side=tk.TOP, fill="x", padx=20)

PCAButton = tk.Button(preproFrame, text="PCA", command=applyPCA)
PCAButton.pack(side=tk.TOP, fill="x", padx=20)

PCAVarPlotButton = tk.Button(preproFrame, text="PCA Variance Plot", command=plotPCAVarPlot)
PCAVarPlotButton.pack(side=tk.TOP, fill="x", padx=20)

PCABiPlotButton = tk.Button(preproFrame, text="PCA BiPlot", command=plotPCABiPlot)
PCABiPlotButton.pack(side=tk.TOP, fill="x", padx=20)






window.mainloop()
