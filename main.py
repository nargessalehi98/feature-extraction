import pandas
import sklearn
from sklearn.decomposition import PCA
import matplotlib.pyplot


def read_data():
    pandas.options.display.max_rows = 160
    data = pandas.read_csv("./iris.data",
                           names=['sepal_length',
                                  'sepal_width',
                                  'petal_length',
                                  'petal_width',
                                  'name'])
    return data


def get_number_of_nan_data():
    data = pandas.DataFrame(read_data())
    sepal_length_missing = 0
    sepal_width_missing = 0
    petal_length_missing = 0
    petal_width_missing = 0
    name_missing = 0
    for index, row in data.iterrows():
        if pandas.isna(row['sepal_length']):
            sepal_length_missing += 1
        if pandas.isna(row['sepal_width']):
            sepal_width_missing += 1
        if pandas.isna(row['petal_length']):
            petal_length_missing += 1
        if pandas.isna(row['petal_width']):
            petal_width_missing += 1
        if pandas.isna(row['name']):
            name_missing += 1
    print("sepal_length_missing: " + str(sepal_length_missing) + "\n" +
          "sepal_width_missing: " + str(sepal_width_missing) + "\n" +
          "petal_width_missing: " + str(petal_width_missing) + "\n"
                                                               "petal_length_missing: " + str(
        petal_length_missing) + "\n"                                                                                                             "name_missing: " + str(
        name_missing) + "\n")


def drop_nan_data(data):
    data = pandas.DataFrame(data)
    data = data.dropna(subset=['name', 'sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    data.reset_index(drop=True, inplace=True)
    return data


def calculate_mean_and_var_and_update_them(data):
    data['name'] = sklearn.preprocessing.LabelEncoder().fit_transform(data['name'])
    name = data.pop('name')
    print("for old data: ")
    sepal_length_avg = data['sepal_length'].mean()
    sepal_width_avg = data["sepal_width"].mean()
    petal_length_avg = data["petal_length"].mean()
    petal_width_avg = data["petal_width"].mean()
    print("sepal_length_avg: " + str(sepal_length_avg))
    print("sepal_width_avg: " + str(sepal_width_avg))
    print("petal_length_avg: " + str(petal_length_avg))
    print("petal_width_avg: " + str(petal_width_avg))
    print("\n")
    sepal_length_var = data['sepal_length'].var()
    sepal_width_var = data["sepal_width"].var()
    petal_length_var = data["petal_length"].var()
    petal_width_var = data["petal_width"].var()
    print("sepal_length_var: " + str(sepal_length_var))
    print("sepal_width_var: " + str(sepal_width_var))
    print("petal_length_var: " + str(petal_length_var))
    print("petal_width_var: " + str(petal_width_var))

    new_data = sklearn.preprocessing.StandardScaler().fit_transform(data)
    new_data = pandas.DataFrame(new_data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

    print("\nfor new data: ")
    sepal_length_avg = new_data['sepal_length'].mean()
    sepal_width_avg = new_data["sepal_width"].mean()
    petal_length_avg = new_data["petal_length"].mean()
    petal_width_avg = new_data["petal_width"].mean()
    print("sepal_length_avg: " + str(sepal_length_avg))
    print("sepal_width_avg: " + str(sepal_width_avg))
    print("petal_length_avg: " + str(petal_length_avg))
    print("petal_width_avg: " + str(petal_width_avg))
    print("\n")
    sepal_length_avg = new_data['sepal_length'].var()
    sepal_width_avg = new_data["sepal_width"].var()
    petal_length_avg = new_data["petal_length"].var()
    petal_width_avg = new_data["petal_width"].var()
    print("sepal_length_var: " + str(sepal_length_avg))
    print("sepal_width_var: " + str(sepal_width_avg))
    print("petal_length_var: " + str(petal_length_avg))
    print("petal_width_var: " + str(petal_width_avg))

    return new_data, name


# OneHotEncoder
# The input to this transformer should be an array-like of integers or strings,
# denoting the values taken on by categorical (discrete) features.
# The features are encoded using a one-hot (aka ‘one-of-K’ or ‘dummy’) encoding scheme.
# This creates a binary column for each category and returns a sparse matrix or dense array (depending on the sparse parameter)
# By default, the encoder derives the categories based on the unique values in each feature.
# Alternatively, you can also specify the categories manually.
# This encoding is needed for feeding categorical data to many scikit-learn estimators,
# notably linear models and SVMs with the standard kernels.
# Examples
##### Given a dataset with two features, we let the encoder
##### find the unique values per feature and transform the data to a binary one-hot encoding.
# >>> from sklearn.preprocessing import OneHotEncoder
# One can discard categories not seen during fit:
# >>> enc = OneHotEncoder(handle_unknown='ignore')
# >>> X = [['Male', 1], ['Female', 3], ['Female', 2]]
# >>> enc.fit(X)
# OneHotEncoder(handle_unknown='ignore')
# >>> enc.categories_
# [array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]
# >>> enc.transform([['Female', 1], ['Male', 4]]).toarray()
# array([[1., 0., 1., 0., 0.],
#        [0., 1., 0., 0., 0.]])
# >>> enc.inverse_transform([[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]])
# array([['Male', 1],
#        [None, 2]], dtype=object)
# >>> enc.get_feature_names_out(['gender', 'group'])
# array(['gender_Female', 'gender_Male', 'group_1', 'group_2', 'group_3'], ...)
##### One can always drop the first column for each feature:
# >>> drop_enc = OneHotEncoder(drop='first').fit(X)
# >>> drop_enc.categories_
# [array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]
# >>> drop_enc.transform([['Female', 1], ['Male', 2]]).toarray()
# array([[0., 0., 0.],
#        [1., 1., 0.]])
##### Or drop a column for feature only having 2 categories:
# >>> drop_binary_enc = OneHotEncoder(drop='if_binary').fit(X)
# >>> drop_binary_enc.transform([['Female', 1], ['Male', 2]]).toarray()
# array([[0., 1., 0., 0.],
#        [1., 0., 1., 0.]])


def four_to_two_D(data):
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(data)
    two_D = pandas.DataFrame(data=principalComponents, columns=['1', '2'])
    return two_D


def draw_data(data, data_2d, name):
    data_2d['color'] = name.map({0: 'Orange', 1: 'Yellow', 2: 'Green'})
    data_2d.plot.scatter(x='1', y='2', c="color")
    matplotlib.pyplot.show()
    data.boxplot(column=['sepal_width', 'sepal_length', 'petal_width', 'petal_length'])
    matplotlib.pyplot.show()


if __name__ == '__main__':
    data = drop_nan_data(read_data())
    get_number_of_nan_data()
    new_data, name = calculate_mean_and_var_and_update_them(data)
    two_d_data = four_to_two_D(new_data)
    draw_data(data, two_d_data, name)
