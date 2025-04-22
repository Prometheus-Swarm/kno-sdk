```python
from repo_indexer import clone_and_index, search

# index a repo
idx = clone_and_index("https://github.com/xyz/project.git")

# later, search
results = search("https://github.com/xyz/project.git", query="How to authenticate?")
for r in results:
    print(r)

---

## 4. Publishing to PyPI

1. **Build** your distribution:

   ```bash
   pip install build twine
   python -m build

2. **Upload** twine upload dist/*
3. **Install** pip install repo-indexer