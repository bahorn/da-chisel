# DA Chisel

This is some experimental algorithms for a graph problem called Defensive
Alliance.
I wrote my Masters thesis on the parameterized complexity of the problem, but
this is just focused on a heuristic solution for the problem as I wanted to see
if applying ML (As done with other optimization problems) could help guide a
branching search.

All the vertices in a graph is always a defensive alliance, which these
algorithms try to "chisel" down by picking vertices and removing them.
The learning goal here was to determine which paths are more likely to result in
smaller alliances.

Currently the model doesn't work well.
Main issue seems to be the scores learned for a given level tend to be far to
similar with not enough distiction.

## Usage

```
mkdir ./data
# generate a bunch of graphs
python3 src gen
# Extract cases from them and find the optimal solution with cbc
python3 src make ./data
# train
python3 src train
# See if we are any better than random
python3 src use
```

You can change various parameters in `src/consts.py`.

## License

MIT
