'''
Model for distributed learning for events using Elephas library.
'''

from pyspark import SparkContext, SparkConf
from elephas.utils.rdd_utils import to_simple_rdd
from elephas.spark_model import SparkModel
from elephas import optimizers as elephas_optimizers


# Parameters:
WORKERS     = 10
EPOCHS      = 20
BATCH_SIZE  = 32
VAL_SPLIT   = 0.1
INPUT_CUT   = 10

# Create a Spark context
conf = SparkConf().setAppName('Elephas').setMaster('local[8]')
sc = SparkContext(conf=conf)

# Keras model

# we feed lists of each particle of length INPUT_CUT.
# The particle types are photons, electrons/positrons, muons/antimuons,
# and neutral, -ve and +ve hadrons

# Particle Number  |  0   |  1   |  2   |  3   | ... |  n   |
# Pt               | pt0  | pt1  | pt2  | pt3  | ... | ptn  |
# Eta              | eta0 | eta1 | eta2 | eta3 | ... | etan |
# Phi              | phi0 | phi1 | phi2 | phi3 | ... | phin |
# Dxy              | dxy0 | dxy1 | dxy2 | dxy3 | ... | dxyn |

# branches for different particle types
def branch():
    model = Sequential()
    model.add(Dense(INPUT_CUT * 4, input_shape=(INPUT_CUT, 4), activation='relu'))
    return model


model = Sequential()

model.add(Merge([branch(), branch(), branch(), branch(), branch()], mode='concat'))

model.add(Dense(30 * INPUT_CUT, activation='relu'))
#model.add(Dropout(0.5))

model.add(Dense(10 * INPUT_CUT, activation='relu'))
#model.add(Dropout(0.5))

model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=SGD())
model.summary()


# Create a Resilient Distributed Dataset (RDD) from training data

# TODO: get data
# TODO: is it possible to separate traininng data into multiple batches?

rdd = to_simple_rdd(sc, X_train, Y_train)


# Create the Elephas model instance
spark_model = SparkModel(sc,
                         model,
                         optimizer = elephas_optimizers.Adagrad(),
                         frequency = 'epoch',
                         mode = 'asynchronous',
                         num_workers = WORKERS
                         )

# Train model
spark_model.train(rdd,
                  nb_epoch = EPOCHS,
                  batch_size = BATCH_SIZE,
                  verbose = False,
                  validation_split = VAL_SPLIT,
                  num_workers = WORKERS
                  )

