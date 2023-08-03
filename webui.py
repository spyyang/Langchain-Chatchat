# 运行方式：
# 1. 安装必要的包：pip install streamlit-option-menu streamlit-chatbox>=1.1.3
# 2. 运行本机fastchat服务：python server\llm_api.py 或者 运行对应的sh文件
# 3. 运行API服务器：python server/api.py。如果使用api = ApiRequest(no_remote_api=True)，该步可以跳过。
# 4. 运行WEB UI：streamlit run webui.py --server.port 7860

import streamlit as st
from webui_pages.utils import *
from streamlit_option_menu import option_menu
from webui_pages import *

api = ApiRequest()

if __name__ == "__main__":
    st.set_page_config("langchain-chatglm WebUI")

    pages = {"对话": {"icon": "chat",
                      "func": dialogue_page,
                      },
             "知识库管理": {"icon": "database-fill-gear",
                            "func": knowledge_base_page,
                            },
             "模型配置": {"icon": "gear",
                          "func": model_config_page,
                          }
             }

    with st.sidebar:
        selected_page = option_menu("langchain-chatglm",
                                    options=list(pages.keys()),
                                    icons=[i["icon"] for i in pages.values()],
                                    menu_icon="chat-quote",
                                    default_index=0)

    pages[selected_page]["func"](api)
