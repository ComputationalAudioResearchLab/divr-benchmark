from .t1 import t1_experiments
from .t2 import t2_experiments
from .t3 import t3_experiments
from .t4 import t4_experiments
from .t5 import t5_experiments

s3_experiments = (
    t1_experiments + t2_experiments + t3_experiments + t4_experiments + t5_experiments
)
