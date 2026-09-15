# 【W6】后端镜像
FROM python:3.12-slim

# 三个环境变量的作用：
#   TZ              容器默认 UTC，不设的话日志时间会比北京时间慢 8 小时
#   PYTHONUNBUFFERED Python 默认缓冲输出，容器里会导致日志不能实时看到
#   PYTHONDONTWRITEBYTECODE 不生成 .pyc，减少镜像层和无用文件
ENV TZ=Asia/Shanghai \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /code

# 先只复制依赖清单：这样以后改业务代码时，pip 安装这一层能命中缓存，
# 重建能从几分钟缩短到几秒。顺序反过来（先 COPY . .）则每次改代码都要重装依赖。
COPY requirements.txt .

# 国内直连 pypi.org 下载 chromadb 这类大包极易超时，所以换清华镜像源。
# 用 ARG 而不是写死：将来在海外环境构建时可覆盖 ——
#   docker compose build --build-arg PIP_INDEX=https://pypi.org/simple
ARG PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --no-cache-dir -i ${PIP_INDEX} -r requirements.txt

COPY . .

EXPOSE 8000

# 健康检查：让编排工具能判断"服务真的可用"而不只是"进程活着"。
# 没有它，docker-compose 的 depends_on 只知道容器起来了，
# 但 FastAPI 可能还在加载模型/建表，此时流量打进来会失败。
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
