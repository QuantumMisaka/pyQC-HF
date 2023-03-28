import numpy as np
# Table by exponents, coefficients, zeta(if not know, default 1)
STO3G_Table = {
    "1s":[
        np.array([2.22766058, 0.40577116, 0.10981751], dtype=np.float64),
        np.array([0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00], dtype=np.float64),
        1.0
    ]
    ,
    "H":  [
        np.array([0.3425250914E+01, 0.6239137298E+00, 0.1688554040E+00], dtype=np.float64),
        np.array([0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00], dtype=np.float64),
        1.24

    ],
    "He": [
        np.array([0.6362421394E+01, 0.1158922999E+01, 0.3136497915E+00], dtype=np.float64),
        np.array([0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00], dtype=np.float64),
        1.69
    ],
    # in Szabo example
    # "He": [
    #     np.array([9.75393717, 1.77669183, 0.48084215], dtype=np.float64),
    #     np.array([0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00], dtype=np.float64),
    #     2.0925
    # ]

    # SP STO-3G support will do in other time -- JamesMisaka in 2023-03-23
    "C": {
        "1s":[
            np.array([0.7161683735E+02, 0.1304509632E+02, 0.3530512160E+01], dtype=np.float64),
            np.array([0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00], dtype=np.float64),
            1
        ],
        "2s2p":[
            np.array([0.2941249355E+01 ,0.6834830964E+00 , 0.1559162750E+00],dtype=np.float64),
            np.array([-0.9996722919E-01 ,0.3995128261E+00 ,0.6076837186E+00],dtype=np.float64),
            np.array([0.2941249355E+01 ,0.7001154689E+00 , 0.3919573931E+00],dtype=np.float64),
        ],
    },

}