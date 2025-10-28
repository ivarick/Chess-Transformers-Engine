Chess engine using the transformers architecture trained on 2 million games (PGN files).
Today's chess engines are shifting to the transformers-based models, we witness that clearly in Leela Chess Zero that switched from residual networks to using a transformer-based architecture.
And this traces back to the limitations of CNNs in chess.
CNNs process information through local convolution operations. Even with many layers and residual connections, they primarily capture local patterns on the chess board. To "see" relationships between pieces on opposite sides of the board requires information to propagate through many layers.
And the huge advantages of Transformers as well. Transformers use self-attention mechanisms that allow every square on the board to directly attend to every other square in a single layer. This is crucial for chess.
A better long range dependecies, chess positions often involve tactics and strategies that connect distant pieces. Transformers excel at modeling these long-range dependencies.
Now the trade-off in using Transformers instead of CNNs is the amount of computation needed, and requiring more memory and processing power. And even a larger dataset.
However, for top-level chess engines, this trade-off is worth it for better positional understanding.

Now in our used model we have:

![alt text](params.png)

When testing the model on finding the best move on the given position:

![alt text](position.png)

Which was a checkmate in one move for black (Ne2#). Chess players recognize this tricky move and it is called a "smothered mate". And yet, our model has successfully found it.

![alt text](results.png)

![alt text](checkmate.png)


