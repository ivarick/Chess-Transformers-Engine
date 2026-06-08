import chess

from features import NUM_FEATURE_PLANES, board_to_features


def test_starting_position_feature_shape_and_piece_count():
    features = board_to_features(chess.Board())

    assert tuple(features.shape) == (NUM_FEATURE_PLANES, 8, 8)
    assert features[:12].sum().item() == 32
    assert features[12].unique().tolist() == [1.0]


def test_en_passant_square_is_encoded():
    board = chess.Board()
    board.push_san("e4")
    board.push_san("a6")
    board.push_san("e5")
    board.push_san("d5")

    features = board_to_features(board)

    assert features[17].sum().item() == 1
