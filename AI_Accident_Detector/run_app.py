import torch
import sys
from streamlit.web import cli as stcli

if __name__ == '__main__':
    # Force PyTorch to initialize its DLLs in the master thread 
    # BEFORE Streamlit is allowed to boot up and hijack the environment
    sys.argv = ["streamlit", "run", "app.py"]
    sys.exit(stcli.main())
