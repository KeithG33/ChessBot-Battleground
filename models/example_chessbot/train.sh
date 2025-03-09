# Train from config, and any overrides in command
chessbot train models/example_chessbot/config.yaml \
              -o model.name=simple_chessbot \
              -o train.epochs=10 \
              -o train.lr=0.001 \
              -o train.batch_size=64 \
              -o dataset.size_train=2 \
              -o dataset.size_test=2 \

