# accelerate launch train.py --cfg_name base
# accelerate launch train.py --cfg_name norm
# accelerate launch train.py --cfg_name coord
accelerate launch train.py --cfg_name normAndCoord
accelerate launch train.py --cfg_name normAndCoordAndSquare
accelerate launch train.py --cfg_name normAndCoordAndMLP2
accelerate launch train.py --cfg_name normAndCoordAndFPN
accelerate launch train.py --cfg_name globalMod