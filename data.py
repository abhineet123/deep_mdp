import paramparse
from utilities import CustomLogger, BaseParams


class Data:
    """
    :type params: Data.Params
    """

    class Params(BaseParams):
        class Datasets:
            gram = [1., 0.]
            idot = [1., 0.]
            detrac = [1., 0.]
            lost = [1., 0.]
            isl = [1., 1.]
            mot2015 = [1., 0.]
            mot2017 = [1., 0.]
            mot2017_sdp = [1., 0.]
            mot2017_dpm = [1., 0.]
            kitti = [1., 0.]
            mnist_mot = [1., 1.]
            mnist_mot_rgb = [1., 1.]
            ctc = [1., 1.]
            ctmc = [1., 1.]
            combined = [1., 1.]

            _names = {
                0: 'MOT2015',
                1: 'MOT2017',
                2: 'MOT2017_SDP',
                3: 'MOT2017_DPM',
                4: 'KITTI',
                5: 'GRAM_ONLY',
                6: 'IDOT',
                7: 'DETRAC',
                8: 'LOST',
                9: 'ISL',
                10: 'GRAM',  # combined sequence set; named GRAM for convenience
                11: 'MNIST_MOT',
                12: 'MNIST_MOT_RGB_128x128_3_25_2000',
                13: 'MNIST_MOT_RGB_512x512_5_25_2000',
                14: 'CTMC',
                15: 'CTC',
            }

            @property
            def names(self): return self._names

        """
        :ivar ratios: 'two element tuple to indicate fraction of frames in each sequence on which'
                  ' (training, testing) is to be performed; '
                  'negative values mean that frames are taken from the end of the sequence; '
                  'zero for the second entry means that all frames not used for training are used for '
                  'testing in each sequence; '
                  'if either entry is > 1, it is set to the corresponding value for the sequence set being used',
        :ivar offsets: 'two element tuple to indicate offsets in the start frame ID with respect to the sub sequence'
                   ' obtained from the (train, test) ratios on which (training, testing) is to be performed;'
                   'ratios and offsets together specify the subsequences, if any, on which the two components'
                   ' of the program are to be run',
        :ivar ratios.gram: 'train and test ratios for sequences in the GRAM dataset',
        :ivar ratios.idot: 'train and test ratios for sequences in the IDOT dataset',
        :ivar ratios.detrac: 'train and test ratios for sequences in the DETRAC dataset',
        :ivar ratios.lost: 'train and test ratios for sequences in the LOST dataset',
        :ivar ratios.isl: 'train and test ratios for sequences in the ISL dataset',
        :ivar ratios.mot2015: 'train and test ratios for sequences in the MOT2015 dataset',
        :ivar ratios.kitti: 'train and test ratios for sequences in the KITTI dataset',

        """

        def __init__(self):
            self.offsets = [0, 0]
            self.ratios = Data.Params.Datasets()

        def synchronize(self, _id=0):
            """
            Equalize training and testing offsets and ratios

            :param _id: 1: Change train offsets and ratios to make them equal to test
            0: Change test offsets and ratios to make them equal to train

            """

            offsets = list(self.offsets)
            offsets[1 - _id] = offsets[_id]
            self.offsets = offsets

            attrs = paramparse.get_valid_members(self.ratios)
            for attr in attrs:
                if attr == 'tee_log':
                    continue
                attr_val = list(getattr(self.ratios, attr))
                attr_val[1 - _id] = attr_val[_id]
                setattr(self.ratios, attr, attr_val)

    def __init__(self, params, logger):
        """
        :type params: Data.Params
        :type logger: logging.RootLogger | CustomLogger
        :rtype: None
        """
        self.params = params
        self._logger = logger
        self.__logger = logger

        self.sets, self.sequences, self.ratios = self.get_sequences()

        self.seq_set = None
        self.seq_name = None
        self.seq_n_frames = 0
        # self.seq_ratios = list(self.params.ratios)

        self.start_frame_id = 0
        self.end_frame_id = 0

        self.is_initialized = False

    def initialize(self, seq_set_id, seq_id, seq_type_id, logger=None, silent=0):
        """
        :type seq_set_id: int
        :type seq_id: int
        :type seq_type_id: int
        :type logger: CustomLogger
        :rtype: bool
        """

        if logger is not None:
            self.__logger = logger

        if seq_set_id < 0 or seq_id < 0:
            self._logger.info('Using external sequence')
            return

        self.seq_set = self.sets[seq_set_id]
        self.seq_name = self.sequences[self.seq_set][seq_id][0]
        self.seq_n_frames = self.sequences[self.seq_set][seq_id][1]

        self._logger = CustomLogger(self.__logger, names=(self.seq_name,), key='custom_header')

        seq_ratios = self.ratios[self.seq_set][seq_id]
        # if self.seq_ratios[0] > 1:
        #     self.seq_ratios[0] = seq_ratios[0]
        # if self.seq_ratios[1] > 1:
        #     self.seq_ratios[1] = seq_ratios[1]

        start_offset = self.params.offsets[seq_type_id]

        if seq_type_id == 0:
            seq_type = 'training'
            seq_ratio = seq_ratios[0]
        else:
            seq_type = 'testing'
            if seq_ratios[1] == 0:
                """
                test on all non-training frames
                """
                if seq_ratios[0] < 0:
                    """training samples from end"""
                    seq_ratio = seq_ratios[0] + 1
                else:
                    seq_ratio = seq_ratios[0] - 1
            else:
                seq_ratio = seq_ratios[1]

        self.start_frame_id, self.end_frame_id = self.get_sub_seq_idx(
            seq_ratio, start_offset, self.seq_n_frames)

        if self.seq_n_frames <= self.start_frame_id or \
                self.start_frame_id < 0 or \
                self.seq_n_frames <= self.end_frame_id or \
                self.end_frame_id <= 0 or \
                self.end_frame_id < self.start_frame_id:
            raise AssertionError('Invalid {:s} ratio: {:.2f} or start_offset:  {:d} '
                                 'specified'.format(seq_type, seq_ratio, start_offset))

        if not silent:
            self._logger.info('seq_ratios: {}'.format(seq_ratios))
            self._logger.info('seq_ratio: {:f}'.format(seq_ratio))
            self._logger.info('start_offset: {:d}'.format(start_offset))

        self.is_initialized = True

        return True

    def get_sub_seq_idx(self, seq_ratio, start_offset, n_frames):
        if seq_ratio < 0:
            """
            sample from end
            """
            start_idx = int(n_frames * (1 + seq_ratio)) - start_offset
            end_idx = int(round(n_frames - start_offset - 1))
        else:
            start_idx = int(start_offset)
            end_idx = int(round(n_frames * seq_ratio)) + start_offset - 1
        if start_idx < 0:
            start_idx = 0
        if end_idx >= n_frames:
            end_idx = n_frames - 1
        return start_idx, end_idx

    def get_inv_sub_seq_idx(self, sub_seq_ratio, start_offset, n_frames):
        if sub_seq_ratio < 0:
            inv_sub_seq_ratio = sub_seq_ratio + 1
        else:
            inv_sub_seq_ratio = sub_seq_ratio - 1
        return self.get_sub_seq_idx(inv_sub_seq_ratio, start_offset, n_frames)

    def combine_sequences(self, seq_lists):
        combined_sequences = {}
        offset = 0
        for sequences in seq_lists:
            for key in sequences:
                combined_sequences[key + offset] = sequences[key]
            offset += len(sequences)
        return combined_sequences

    def get_sequences(self):
        # names of sequences and the no. of frames in each

        sequences = Data.Params.Datasets()

        sequences.mot2015 = {
            # train
            0: ('ADL-Rundle-6', 525),
            1: ('ADL-Rundle-8', 654),
            2: ('ETH-Bahnhof', 1000),
            3: ('ETH-Pedcross2', 837),
            4: ('ETH-Sunnyday', 354),
            5: ('KITTI-13', 340),
            6: ('KITTI-17', 145),
            7: ('PETS09-S2L1', 795),
            8: ('TUD-Campus', 71),
            9: ('TUD-Stadtmitte', 179),
            10: ('Venice-2', 600),
            # test
            11: ('ADL-Rundle-1', 500),
            12: ('ADL-Rundle-3', 625),
            13: ('AVG-TownCentre', 450),
            14: ('ETH-Crossing', 219),
            15: ('ETH-Jelmoli', 440),
            16: ('ETH-Linthescher', 1194),
            17: ('KITTI-16', 209),
            18: ('KITTI-19', 1059),
            19: ('PETS09-S2L2', 436),
            20: ('TUD-Crossing', 201),
            21: ('Venice-1', 450),
        }
        sequences.mot2017 = {
            # train
            0: ('MOT17-02-FRCNN', 600),
            1: ('MOT17-04-FRCNN', 1050),
            2: ('MOT17-05-FRCNN', 837),
            3: ('MOT17-09-FRCNN', 525),
            4: ('MOT17-10-FRCNN', 654),
            5: ('MOT17-11-FRCNN', 900),
            6: ('MOT17-13-FRCNN', 750),
            # test
            7: ('MOT17-01-FRCNN', 450),
            8: ('MOT17-03-FRCNN', 1500),
            9: ('MOT17-06-FRCNN', 1194),
            10: ('MOT17-07-FRCNN', 500),
            11: ('MOT17-08-FRCNN', 625),
            12: ('MOT17-12-FRCNN', 900),
            13: ('MOT17-14-FRCNN', 750),
        }
        sequences.mot2017_sdp = {
            # train
            0: ('MOT17-02-SDP', 600),
            1: ('MOT17-04-SDP', 1050),
            2: ('MOT17-05-SDP', 837),
            3: ('MOT17-09-SDP', 525),
            4: ('MOT17-10-SDP', 654),
            5: ('MOT17-11-SDP', 900),
            6: ('MOT17-13-SDP', 750),
            # test
            7: ('MOT17-01-SDP', 450),
            8: ('MOT17-03-SDP', 1500),
            9: ('MOT17-06-SDP', 1194),
            10: ('MOT17-07-SDP', 500),
            11: ('MOT17-08-SDP', 625),
            12: ('MOT17-12-SDP', 900),
            13: ('MOT17-14-SDP', 750),
        }

        sequences.mot2017_dpm = {
            # train
            0: ('MOT17-02-DPM', 600),
            1: ('MOT17-04-DPM', 1050),
            2: ('MOT17-05-DPM', 837),
            3: ('MOT17-09-DPM', 525),
            4: ('MOT17-10-DPM', 654),
            5: ('MOT17-11-DPM', 900),
            6: ('MOT17-13-DPM', 750),
            # test
            7: ('MOT17-01-DPM', 450),
            8: ('MOT17-03-DPM', 1500),
            9: ('MOT17-06-DPM', 1194),
            10: ('MOT17-07-DPM', 500),
            11: ('MOT17-08-DPM', 625),
            12: ('MOT17-12-DPM', 900),
            13: ('MOT17-14-DPM', 750),
        }

        sequences.kitti = {
            0: ('train_0000', 154),
            1: ('train_0001', 447),
            2: ('train_0002', 233),
            3: ('train_0003', 144),
            4: ('train_0004', 314),
            5: ('train_0005', 297),
            6: ('train_0006', 270),
            7: ('train_0007', 800),
            8: ('train_0008', 390),
            9: ('train_0009', 803),
            10: ('train_0010', 294),
            11: ('train_0011', 373),
            12: ('train_0012', 78),
            13: ('train_0013', 340),
            14: ('train_0014', 106),
            15: ('train_0015', 376),
            16: ('train_0016', 209),
            17: ('train_0017', 145),
            18: ('train_0018', 339),
            19: ('train_0019', 1059),
            20: ('train_0020', 837),
            21: ('test_0000', 465),
            22: ('test_0001', 147),
            23: ('test_0002', 243),
            24: ('test_0003', 257),
            25: ('test_0004', 421),
            26: ('test_0005', 809),
            27: ('test_0006', 114),
            28: ('test_0007', 215),
            29: ('test_0008', 165),
            30: ('test_0009', 349),
            31: ('test_0010', 1176),
            32: ('test_0011', 774),
            33: ('test_0012', 694),
            34: ('test_0013', 152),
            35: ('test_0014', 850),
            36: ('test_0015', 701),
            37: ('test_0016', 510),
            38: ('test_0017', 305),
            39: ('test_0018', 180),
            40: ('test_0019', 404),
            41: ('test_0020', 173),
            42: ('test_0021', 203),
            43: ('test_0022', 436),
            44: ('test_0023', 430),
            45: ('test_0024', 316),
            46: ('test_0025', 176),
            47: ('test_0026', 170),
            48: ('test_0027', 85),
            49: ('test_0028', 175)
        }
        sequences.gram = {
            0: ('M-30', 7520),
            1: ('M-30-HD', 9390),
            2: ('Urban1', 23435),
            # 3: ('M-30-Large', 7520),
            # 4: ('M-30-HD-Small', 9390)
        }
        sequences.idot = {
            0: ('idot_1_intersection_city_day', 8991),
            1: ('idot_2_intersection_suburbs', 8990),
            2: ('idot_3_highway', 8981),
            3: ('idot_4_intersection_city_day', 8866),
            4: ('idot_5_intersection_suburbs', 8851),
            5: ('idot_6_highway', 8791),
            6: ('idot_7_intersection_city_night', 8964),
            7: ('idot_8_intersection_city_night', 8962),
            8: ('idot_9_intersection_city_day', 8966),
            9: ('idot_10_city_road', 7500),
            10: ('idot_11_highway', 7500),
            11: ('idot_12_city_road', 7500),
            12: ('idot_13_intersection_city_day', 8851),
            # 13: ('idot_1_intersection_city_day_short', 84)
        }
        sequences.lost = {
            0: ('009_2011-03-23_07-00-00', 3027),
            1: ('009_2011-03-24_07-00-00', 5000)
        }
        sequences.isl = {
            0: ('isl_1_20170620-055940', 10162),
            1: ('isl_2_20170620-060941', 10191),
            2: ('isl_3_20170620-061942', 10081),
            3: ('isl_4_20170620-062943', 10089),
            4: ('isl_5_20170620-063943', 10177),
            5: ('isl_6_20170620-064944', 10195),
            6: ('isl_7_20170620-065944', 10167),
            7: ('isl_8_20170620-070945', 10183),
            8: ('isl_9_20170620-071946', 10174),
            9: ('isl_10_20170620-072946', 10127),
            10: ('isl_11_20170620-073947', 9738),
            11: ('isl_12_20170620-074947', 10087),
            12: ('isl_13_20170620-075949', 8614),
            13: ('ISL16F8J_TMC_SCU2DJ_2016-10-05_0700', 1188000),
            14: ('DJI_0020_masked_2000', 2000),
            15: ('debug_with_colors', 10)
        }
        sequences.detrac = {
            # train
            0: ('detrac_1_MVI_20011', 664),
            1: ('detrac_2_MVI_20012', 936),
            2: ('detrac_3_MVI_20032', 437),
            3: ('detrac_4_MVI_20033', 784),
            4: ('detrac_5_MVI_20034', 800),
            5: ('detrac_6_MVI_20035', 800),
            6: ('detrac_7_MVI_20051', 906),
            7: ('detrac_8_MVI_20052', 694),
            8: ('detrac_9_MVI_20061', 800),
            9: ('detrac_10_MVI_20062', 800),
            10: ('detrac_11_MVI_20063', 800),
            11: ('detrac_12_MVI_20064', 800),
            12: ('detrac_13_MVI_20065', 1200),
            13: ('detrac_14_MVI_39761', 1660),
            14: ('detrac_15_MVI_39771', 570),
            15: ('detrac_16_MVI_39781', 1865),
            16: ('detrac_17_MVI_39801', 885),
            17: ('detrac_18_MVI_39811', 1070),
            18: ('detrac_19_MVI_39821', 880),
            19: ('detrac_20_MVI_39851', 1420),
            20: ('detrac_21_MVI_39861', 745),
            21: ('detrac_22_MVI_39931', 1270),
            22: ('detrac_23_MVI_40131', 1645),
            23: ('detrac_24_MVI_40141', 1600),
            24: ('detrac_25_MVI_40152', 1750),
            25: ('detrac_26_MVI_40161', 1490),
            26: ('detrac_27_MVI_40162', 1765),
            27: ('detrac_28_MVI_40171', 1150),
            28: ('detrac_29_MVI_40172', 2635),
            29: ('detrac_30_MVI_40181', 1700),
            30: ('detrac_31_MVI_40191', 2495),
            31: ('detrac_32_MVI_40192', 2195),
            32: ('detrac_33_MVI_40201', 925),
            33: ('detrac_34_MVI_40204', 1225),
            34: ('detrac_35_MVI_40211', 1950),
            35: ('detrac_36_MVI_40212', 1690),
            36: ('detrac_37_MVI_40213', 1790),
            37: ('detrac_38_MVI_40241', 2320),
            38: ('detrac_39_MVI_40243', 1265),
            39: ('detrac_40_MVI_40244', 1345),
            40: ('detrac_41_MVI_40732', 2120),
            41: ('detrac_42_MVI_40751', 1145),
            42: ('detrac_43_MVI_40752', 2025),
            43: ('detrac_44_MVI_40871', 1720),
            44: ('detrac_45_MVI_40962', 1875),
            45: ('detrac_46_MVI_40963', 1820),
            46: ('detrac_47_MVI_40981', 1995),
            47: ('detrac_48_MVI_40991', 1820),
            48: ('detrac_49_MVI_40992', 2160),
            49: ('detrac_50_MVI_41063', 1505),
            50: ('detrac_51_MVI_41073', 1825),
            51: ('detrac_52_MVI_63521', 2055),
            52: ('detrac_53_MVI_63525', 985),
            53: ('detrac_54_MVI_63544', 1160),
            54: ('detrac_55_MVI_63552', 1150),
            55: ('detrac_56_MVI_63553', 1405),
            56: ('detrac_57_MVI_63554', 1445),
            57: ('detrac_58_MVI_63561', 1285),
            58: ('detrac_59_MVI_63562', 1185),
            59: ('detrac_60_MVI_63563', 1390),
            # test
            60: ('detrac_61_MVI_39031', 1470),
            61: ('detrac_62_MVI_39051', 1120),
            62: ('detrac_63_MVI_39211', 1660),
            63: ('detrac_64_MVI_39271', 1570),
            64: ('detrac_65_MVI_39311', 1505),
            65: ('detrac_66_MVI_39361', 2030),
            66: ('detrac_67_MVI_39371', 1390),
            67: ('detrac_68_MVI_39401', 1385),
            68: ('detrac_69_MVI_39501', 540),
            69: ('detrac_70_MVI_39511', 380),
            70: ('detrac_71_MVI_40701', 1130),
            71: ('detrac_72_MVI_40711', 1030),
            72: ('detrac_73_MVI_40712', 2400),
            73: ('detrac_74_MVI_40714', 1180),
            74: ('detrac_75_MVI_40742', 1655),
            75: ('detrac_76_MVI_40743', 1630),
            76: ('detrac_77_MVI_40761', 2030),
            77: ('detrac_78_MVI_40762', 1825),
            78: ('detrac_79_MVI_40763', 1745),
            79: ('detrac_80_MVI_40771', 1720),
            80: ('detrac_81_MVI_40772', 1200),
            81: ('detrac_82_MVI_40773', 985),
            82: ('detrac_83_MVI_40774', 950),
            83: ('detrac_84_MVI_40775', 975),
            84: ('detrac_85_MVI_40792', 1810),
            85: ('detrac_86_MVI_40793', 1960),
            86: ('detrac_87_MVI_40851', 1140),
            87: ('detrac_88_MVI_40852', 1150),
            88: ('detrac_89_MVI_40853', 1590),
            89: ('detrac_90_MVI_40854', 1195),
            90: ('detrac_91_MVI_40855', 1090),
            91: ('detrac_92_MVI_40863', 1670),
            92: ('detrac_93_MVI_40864', 1515),
            93: ('detrac_94_MVI_40891', 1545),
            94: ('detrac_95_MVI_40892', 1790),
            95: ('detrac_96_MVI_40901', 1335),
            96: ('detrac_97_MVI_40902', 1005),
            97: ('detrac_98_MVI_40903', 1060),
            98: ('detrac_99_MVI_40904', 1270),
            99: ('detrac_100_MVI_40905', 1710),
        }

        sequences.mnist_mot = {
            # train
            0: ('train_0', 1968),
            1: ('train_1', 1989),
            2: ('train_2', 1994),
            3: ('train_3', 1958),
            4: ('train_4', 1895),
            5: ('train_5', 1962),
            6: ('train_6', 1959),
            7: ('train_7', 1928),
            8: ('train_8', 1991),
            9: ('train_9', 1946),
            10: ('train_10', 1994),
            11: ('train_11', 1982),
            12: ('train_12', 1957),
            13: ('train_13', 1999),
            14: ('train_14', 1964),
            15: ('train_15', 1976),
            16: ('train_16', 1904),
            17: ('train_17', 1913),
            18: ('train_18', 1942),
            19: ('train_19', 1929),
            20: ('train_20', 1982),
            21: ('train_21', 1913),
            22: ('train_22', 1988),
            23: ('train_23', 1890),
            24: ('train_24', 1984),
            # test
            25: ('test_0', 1965),
            26: ('test_1', 1952),
            27: ('test_2', 1938),
            28: ('test_3', 1941),
            29: ('test_4', 1981),
            30: ('test_5', 1941),
            31: ('test_6', 1969),
            32: ('test_7', 1981),
            33: ('test_8', 1959),
            34: ('test_9', 1974),
            35: ('test_10', 1929),
            36: ('test_11', 1999),
            37: ('test_12', 1957),
            38: ('test_13', 1928),
            39: ('test_14', 1976),
            40: ('test_15', 1968),
            41: ('test_16', 2000),
            42: ('test_17', 1998),
            43: ('test_18', 1998),
            44: ('test_19', 1977),
            45: ('test_20', 1923),
            46: ('test_21', 1971),
            47: ('test_22', 1973),
            48: ('test_23', 1992),
            49: ('test_24', 1980),
        }
        sequences.mnist_mot_rgb_128 = {
            0: ('train_0_light_pink_255_182_193', 1910),
            1: ('train_1_medium_sea_green_60_179_113', 1913),
            2: ('train_2_navy_0_0_128', 1937),
            3: ('train_3_bisque_2_238_213_183', 1994),
            4: ('train_4_old_lace_253_245_230', 1965),
            5: ('train_5_snow_2_238_233_233', 1927),
            6: ('train_6_goldenrod_218_165_32', 1973),
            7: ('train_7_dark_turquoise_0_206_209', 1998),
            8: ('train_8_plum_221_160_221', 1917),
            9: ('train_9_dim_gray_105_105_105', 1990),
            10: ('train_10_dim_gray_105_105_105', 1998),
            11: ('train_11_medium_blue_0_0_205', 1947),
            12: ('train_12_medium_violet_red_199_21_133', 1986),
            13: ('train_13_green_yellow_173_255_47', 1985),
            14: ('train_14_pale_goldenrod_238_232_170', 1985),
            15: ('train_15_rosy_brown_188_143_143', 1998),
            16: ('train_16_sienna_160_82_45', 1959),
            17: ('train_17_cornsilk_2_238_232_205', 1973),
            18: ('train_18_cornsilk_2_238_232_205', 1943),
            19: ('train_19_ghost_white_248_248_255', 1951),
            20: ('train_20_seashell_255_245_238', 1985),
            21: ('train_21_peru_205_133_63', 1937),
            22: ('train_22_goldenrod_218_165_32', 1906),
            23: ('train_23_royal_blue_65_105_225', 1953),
            24: ('train_24_peru_205_133_63', 1958),
            25: ('test_0_pale_violet_red_219_112_147', 1962),
            26: ('test_1_thistle_216_191_216', 1958),
            27: ('test_2_burlywood_222_184_135', 1966),
            28: ('test_3_light_sky_blue_135_206_250', 1927),
            29: ('test_4_ivory_255_255_240', 1954),
            30: ('test_5_dark_goldenrod_184_134_11', 1938),
            31: ('test_6_violet_238_130_238', 1920),
            32: ('test_7_green_yellow_173_255_47', 1951),
            33: ('test_8_medium_purple_147_112_219', 1959),
            34: ('test_9_medium_turquoise_72_209_204', 1989),
            35: ('test_10_cadet_blue_95_158_160', 1914),
            36: ('test_11_light_sea_green_32_178_170', 1893),
            37: ('test_12_light_coral_240_128_128', 1997),
            38: ('test_13_rosy_brown_188_143_143', 1967),
            39: ('test_14_peach_puff_3_205_175_149', 1945),
            40: ('test_15_maroon_176_48_96', 1937),
            41: ('test_16_light_salmon_255_160_122', 1987),
            42: ('test_17_dark_turquoise_0_206_209', 1964),
            43: ('test_18_medium_slate_blue_123_104_238', 1931),
            44: ('test_19_tomato_255_99_71', 1937),
            45: ('test_20_saddle_brown_139_69_19', 1989),
            46: ('test_21_goldenrod_218_165_32', 1994),
            47: ('test_22_blue_0_0_255', 1949),
            48: ('test_23_cadet_blue_95_158_160', 1976),
            49: ('test_24_medium_aquamarine_102_205_170', 1996)
        }
        sequences.ctc = {
            # train
            0: ('BF-C2DL-HSC_01', 1764),
            1: ('BF-C2DL-HSC_02', 1764),
            2: ('BF-C2DL-MuSC_01', 1376),
            3: ('BF-C2DL-MuSC_02', 1376),

            4: ('DIC-C2DH-HeLa_01', 84),
            5: ('DIC-C2DH-HeLa_02', 84),

            6: ('Fluo-C2DL-Huh7_01', 30),
            7: ('Fluo-C2DL-Huh7_02', 30),

            8: ('Fluo-N2DH-GOWT1_01', 92),
            9: ('Fluo-N2DH-GOWT1_02', 92),

            10: ('Fluo-N2DH-SIM_01', 65),
            11: ('Fluo-N2DH-SIM_02', 150),

            12: ('Fluo-C2DL-MSC_01', 48),
            13: ('Fluo-C2DL-MSC_02', 48),

            14: ('Fluo-N2DL-HeLa_01', 92),
            15: ('Fluo-N2DL-HeLa_02', 92),

            16: ('PhC-C2DH-U373_01', 115),
            17: ('PhC-C2DH-U373_02', 115),
            18: ('PhC-C2DL-PSC_01', 300),
            19: ('PhC-C2DL-PSC_02', 300),

            # test

            20: ('BF-C2DL-HSC_Test_01', 1764),
            21: ('BF-C2DL-HSC_Test_02', 1764),
            22: ('BF-C2DL-MuSC_Test_01', 1376),
            23: ('BF-C2DL-MuSC_Test_02', 1376),

            24: ('DIC-C2DH-HeLa_Test_01', 115),
            25: ('DIC-C2DH-HeLa_Test_02', 115),

            26: ('Fluo-C2DL-Huh7_Test_01', 30),
            27: ('Fluo-C2DL-Huh7_Test_02', 30),

            28: ('Fluo-N2DH-GOWT1_Test_01', 92),
            29: ('Fluo-N2DH-GOWT1_Test_02', 92),

            30: ('Fluo-N2DH-SIM_Test_01', 110),
            31: ('Fluo-N2DH-SIM_Test_02', 138),

            32: ('Fluo-C2DL-MSC_Test_01', 48),
            33: ('Fluo-C2DL-MSC_Test_02', 48),

            34: ('Fluo-N2DL-HeLa_Test_01', 92),
            35: ('Fluo-N2DL-HeLa_Test_02', 92),

            36: ('PhC-C2DH-U373_Test_01', 115),
            37: ('PhC-C2DH-U373_Test_02', 115),
            38: ('PhC-C2DL-PSC_Test_02', 300),
            39: ('PhC-C2DL-PSC_Test_01', 300),
        }
        sequences.ctmc = {
            # train
            0: ('3T3-run01', 1770),
            1: ('3T3-run03', 2062),
            2: ('3T3-run05', 2039),
            3: ('3T3-run07', 1931),
            4: ('3T3-run09', 2114),
            5: ('A-10-run01', 1969),
            6: ('A-10-run03', 1891),
            7: ('A-10-run05', 1571),
            8: ('A-10-run07', 2462),
            9: ('A-549-run03', 1592),
            10: ('APM-run01', 1794),
            11: ('APM-run03', 1458),
            12: ('APM-run05', 2006),
            13: ('BPAE-run01', 1568),
            14: ('BPAE-run03', 1803),
            15: ('BPAE-run05', 1775),
            16: ('BPAE-run07', 2160),
            17: ('CRE-BAG2-run01', 1931),
            18: ('CRE-BAG2-run03', 1652),
            19: ('CV-1-run01', 1842),
            20: ('CV-1-run03', 1613),
            21: ('LLC-MK2-run01', 1866),
            22: ('LLC-MK2-run02a', 1050),
            23: ('LLC-MK2-run03', 2040),
            24: ('LLC-MK2-run05', 1861),
            25: ('LLC-MK2-run07', 2187),
            26: ('MDBK-run01', 368),
            27: ('MDBK-run03', 780),
            28: ('MDBK-run05', 1500),
            29: ('MDBK-run07', 1824),
            30: ('MDBK-run09', 2195),
            31: ('MDOK-run01', 2100),
            32: ('MDOK-run03', 763),
            33: ('MDOK-run05', 1894),
            34: ('MDOK-run07', 1972),
            35: ('MDOK-run09', 2073),
            36: ('OK-run01', 1200),
            37: ('OK-run03', 1350),
            38: ('OK-run05', 1350),
            39: ('OK-run07', 1620),
            40: ('PL1Ut-run01', 1731),
            41: ('PL1Ut-run03', 1848),
            42: ('PL1Ut-run05', 1470),
            43: ('RK-13-run01', 1500),
            44: ('RK-13-run03', 900),
            45: ('U2O-S-run03', 1972),
            46: ('U2O-S-run05', 1972),
            # test
            47: ('3T3-run02', 2135),
            48: ('3T3-run04', 2160),
            49: ('3T3-run06', 1680),
            50: ('3T3-run08', 1350),
            51: ('A-10-run02', 2164),
            52: ('A-10-run04', 1200),
            53: ('A-10-run06', 1890),
            54: ('A-549-run02', 2284),
            55: ('A-549-run04', 1881),
            56: ('APM-run02', 2069),
            57: ('APM-run04', 2119),
            58: ('APM-run06', 2382),
            59: ('BPAE-run02', 2154),
            60: ('BPAE-run04', 1905),
            61: ('BPAE-run06', 2119),
            62: ('CRE-BAG2-run02', 2543),
            63: ('CRE-BAG2-run04', 1827),
            64: ('CV-1-run02', 293),
            65: ('CV-1-run04', 1050),
            66: ('LLC-MK2-run02b', 839),
            67: ('LLC-MK2-run04', 1894),
            68: ('LLC-MK2-run06', 1716),
            69: ('MDBK-run02', 583),
            70: ('MDBK-run04', 2046),
            71: ('MDBK-run06', 1717),
            72: ('MDBK-run08', 2012),
            73: ('MDBK-run10', 2402),
            74: ('MDOK-run02', 2001),
            75: ('MDOK-run04', 1618),
            76: ('MDOK-run06', 2364),
            77: ('MDOK-run08', 1765),
            78: ('OK-run02', 2110),
            79: ('OK-run04', 900),
            80: ('OK-run06', 1912),
            81: ('PL1Ut-run02', 1350),
            82: ('PL1Ut-run04', 2062),
            83: ('RK-13-run02', 1050),
            84: ('U2O-S-run02', 4437),
            85: ('U2O-S-run04', 2126),

        }
        sequences.mnist_mot_rgb_512 = {
            0: ('train_0_antique_white_2_238_223_204', 1991),
            1: ('train_1_lemon_chiffon_255_250_205', 2000),
            2: ('train_2_olive_drab_107_142_35', 1997),
            3: ('train_3_misty_rose_255_228_225', 2000),
            4: ('train_4_green_0_255_0', 2000),
            5: ('train_5_ivory_255_255_240', 2000),
            6: ('train_6_dark_orchid_153_50_204', 1999),
            7: ('train_7_ivory_4_139_139_131', 2000),
            8: ('train_8_snow_3_205_201_201', 2000),
            9: ('train_9_navy_0_0_128', 2000),
            10: ('train_10_cornflower_blue_100_149_237', 2000),
            11: ('train_11_cornsilk_2_238_232_205', 2000),
            12: ('train_12_cornsilk_2_238_232_205', 2000),
            13: ('train_13_salmon_250_128_114', 2000),
            14: ('train_14_azure_240_255_255', 2000),
            15: ('train_15_light_gray_211_211_211', 2000),
            16: ('train_16_seashell_255_245_238', 2000),
            17: ('train_17_pink_255_192_203', 2000),
            18: ('train_18_medium_turquoise_72_209_204', 2000),
            19: ('train_19_burlywood_222_184_135', 2000),
            20: ('train_20_seashell_4_139_134_130', 2000),
            21: ('train_21_cornsilk_255_248_220', 2000),
            22: ('train_22_yellow_255_255_0', 2000),
            23: ('train_23_dodger_blue_30_144_255', 2000),
            24: ('train_24_green_0_255_0', 2000),
            25: ('test_0_antique_white_250_235_215', 2000),
            26: ('test_1_sandy_brown_244_164_96', 2000),
            27: ('test_2_lavender_230_230_250', 2000),
            28: ('test_3_light_pink_255_182_193', 2000),
            29: ('test_4_orchid_218_112_214', 2000),
            30: ('test_5_dark_goldenrod_184_134_11', 2000),
            31: ('test_6_light_slate_gray_119_136_153', 2000),
            32: ('test_7_honeydew_3_193_205_193', 2000),
            33: ('test_8_cornsilk_4_139_136_120', 2000),
            34: ('test_9_light_sky_blue_135_206_250', 2000),
            35: ('test_10_honeydew_4_131_139_131', 2000),
            36: ('test_11_midnight_blue_25_25_112', 2000),
            37: ('test_12_light_sea_green_32_178_170', 2000),
            38: ('test_13_lemon_chiffon_255_250_205', 2000),
            39: ('test_14_dark_orange_255_140_0', 2000),
            40: ('test_15_dark_violet_148_0_211', 2000),
            41: ('test_16_seashell_3_205_197_191', 2000),
            42: ('test_17_bisque_2_238_213_183', 2000),
            43: ('test_18_seashell_3_205_197_191', 2000),
            44: ('test_19_sky_blue_135_206_250', 2000),
            45: ('test_20_medium_sea_green_60_179_113', 2000),
            46: ('test_21_plum_221_160_221', 2000),
            47: ('test_22_ivory_3_205_205_193', 2000),
            48: ('test_23_antique_white_3_205_192_176', 2000),
            49: ('test_24_medium_purple_147_112_219', 2000)
        }

        # combined GRAM, IDOT, LOST and ISL for convenience of inter-dataset training and testing
        sequences.combined = self.combine_sequences((
            sequences.gram,  # 0 - 2
            sequences.idot,  # 3 - 15
            # sequences.detrac,  # 17 - 116
            # sequences.lost,  # 117 - 118
            # sequences.isl  # 120 - 135
        ))

        sets = sequences.names

        sequences_dict = dict(zip([sets[i] for i in range(len(sets))],
                                  [sequences.mot2015,
                                   sequences.mot2017,
                                   sequences.mot2017_sdp,
                                   sequences.mot2017_dpm,
                                   sequences.kitti,
                                   sequences.gram,
                                   sequences.idot,
                                   sequences.detrac,
                                   sequences.lost,
                                   sequences.isl,
                                   sequences.combined,
                                   sequences.mnist_mot,
                                   sequences.mnist_mot_rgb_128,
                                   sequences.mnist_mot_rgb_512,
                                   sequences.ctmc,
                                   sequences.ctc,
                                   ]))

        ratios = Data.Params.Datasets()

        ratios.mot2015 = dict(zip(range(len(sequences.mot2015)),
                                  [self.params.ratios.mot2015] * len(sequences.mot2015)))
        ratios.mot2017 = dict(zip(range(len(sequences.mot2017)),
                                  [self.params.ratios.mot2017] * len(sequences.mot2017)))
        ratios.mot2017_sdp = dict(zip(range(len(sequences.mot2017_sdp)),
                                      [self.params.ratios.mot2017_sdp] * len(sequences.mot2017_sdp)))
        ratios.mot2017_dpm = dict(zip(range(len(sequences.mot2017_dpm)),
                                      [self.params.ratios.mot2017_dpm] * len(sequences.mot2017_dpm)))

        ratios.kitti = dict(zip(range(len(sequences.kitti)),
                                [self.params.ratios.kitti] * len(sequences.kitti)))
        ratios.gram = dict(zip(range(len(sequences.gram)),
                               [self.params.ratios.gram] * len(sequences.gram)))
        ratios.idot = dict(zip(range(len(sequences.idot)),
                               [self.params.ratios.idot] * len(sequences.idot)))
        ratios.detrac = dict(zip(range(len(sequences.detrac)),
                                 [self.params.ratios.detrac] * len(sequences.detrac)))
        ratios.lost = dict(zip(range(len(sequences.lost)),
                               [self.params.ratios.lost] * len(sequences.lost)))
        ratios.isl = dict(zip(range(len(sequences.isl)),
                              [self.params.ratios.isl] * len(sequences.isl)))
        ratios.mnist_mot = dict(zip(range(len(sequences.mnist_mot)),
                                    [self.params.ratios.mnist_mot] * len(sequences.mnist_mot)))

        ratios.mnist_mot_rgb = dict(zip(range(len(sequences.mnist_mot)),
                                        [self.params.ratios.mnist_mot_rgb] * len(sequences.mnist_mot_rgb_128)))

        ratios.ctmc = dict(zip(range(len(sequences.ctmc)),
                               [self.params.ratios.ctmc] * len(sequences.ctmc)))

        ratios.ctc = dict(zip(range(len(sequences.ctc)),
                               [self.params.ratios.ctc] * len(sequences.ctc)))

        ratios.combined = self.combine_sequences((
            ratios.gram,  # 0 - 2
            ratios.idot,  # 3 - 15
            # ratios.detrac,  # 19 - 78
            # ratios.lost,  # 79 - 80
            # ratios.isl  # 81 - 95
        ))
        ratios_dict = dict(zip([sets[i] for i in range(len(sets))],
                               [ratios.mot2015,
                                ratios.mot2017,
                                ratios.mot2017_sdp,
                                ratios.mot2017_dpm,
                                ratios.kitti,
                                ratios.gram,
                                ratios.idot,
                                ratios.detrac,
                                ratios.lost,
                                ratios.isl,
                                ratios.combined,
                                ratios.mnist_mot,
                                ratios.mnist_mot_rgb,
                                ratios.mnist_mot_rgb,
                                ratios.ctmc,
                                ratios.ctc,
                                ]))

        return sets, sequences_dict, ratios_dict
