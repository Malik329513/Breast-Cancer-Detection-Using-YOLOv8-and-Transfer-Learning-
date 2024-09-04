# my_project/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns

def show_image(image, cmap='gray'):
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    plt.show()

def plot_correlation_matrix(df, graphWidth):
    corr = df.corr()
    plt.figure(figsize=(graphWidth, graphWidth))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.show()

def plot_histogram(df, nGraphShown, nGraphPerRow):
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = int(np.ceil(nCol / nGraphPerRow))
    plt.figure(figsize=(6 * nGraphPerRow, 8 * nGraphRow))
    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if np.issubdtype(type(columnDf.iloc[0]), np.number):
            columnDf.hist()
            plt.title(f'{columnNames[i]}')
    plt.show()
