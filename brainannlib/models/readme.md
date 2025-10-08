

## How were the models downloaded to the package?

### BEATs

```sh
cd models/
git clone --filter=blob:none --no-checkout https://github.com/microsoft/unilm
cd unilm
git sparse-checkout init --cone
git sparse-checkout set beats
git checkout
cd ..
mv unilm/beats/ beats
cd beats/
touch __init__.py
```

Then I had to additionally add a "." before the local imports in the `backbone.py`, `BEATs.py` and `Tokenizers.py`, e.g.:

```python
# before: 
# from backbone import (TransformerEncoder,)
# after:
from .backbone import (TransformerEncoder,)
```