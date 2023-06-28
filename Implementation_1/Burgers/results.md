# Result 1 [no adaptation]

### Number of points for interior, boundary and inital condition
Nr = int(960)
Nb = int(320) 
Ni = int(320)

### Batch size
Br = 320
Bb = 320
Bi = 320

neurons = [400, 400, 400, 400, 400]
compute_NTK_interval = 1000

learning_rate = 1e-5
epochs        = int(80e3)

use_adaptation_algorithm = False
adapt_lr                 = False

Adam optimizer


# Result 2 [adaptation]

### Number of points for interior, boundary and inital condition
Nr = int(960)
Nb = int(320) 
Ni = int(320)

### Batch size
Br = 320
Bb = 320
Bi = 320

neurons = [400, 400, 400, 400, 400]
compute_NTK_interval = 1000

learning_rate = 1e-5
epochs        = int(80e3)

use_adaptation_algorithm = True
adapt_lr                 = False

Adam optimizer

# Result 3 [adaptation with mixed batch sizes]

### Number of points for interior, boundary and inital condition
Nr = int(960)
Nb = int(64) 
Ni = int(128)

### Batch size
Br = 320
Bb = 64
Bi = 128

neurons = [400, 400, 400, 400, 400]
learning_rate = 1e-5
epochs        = int(40e3)

compute_NTK_interval = 1000
use_adaptation_algorithm = True
adapt_lr                 = False

Adam optimizer

# Result 4 [optimal]

### Number of points for interior, boundary and inital condition
Nr = int(3200)
Nb = int(160) 
Ni = int(320)

### Batch size
Br = 640
Bb = 80
Bi = 160

neurons = [480, 480, 480, 480, 480]
learning_rate = 1e-3
epochs        = int(40e3)

compute_NTK_interval = 1000
use_adaptation_algorithm = True
adapt_lr                 = True

Adam optimizer


# Pitfalls

* Balancing act between suffient set of datapoints and batch size. If the batch size or dataset is too small, the adaptation algorithm can have negative weights resulting in unstable training behavior.