# Elasticsearch 8.15.0 + IK 中文分词器
# 用 Dockerfile 装 IK，好处是容器重建后插件还在（docker exec 装的重建就没了）
FROM elasticsearch:8.15.0

# IK 分词器（INFINI 官方分发地址，8.15.0 对应版本）
RUN bin/elasticsearch-plugin install --batch \
    https://get.infini.cloud/elasticsearch/analysis-ik/8.15.0

# 校验插件确实装上了（构建失败比运行时报错好排查）
RUN bin/elasticsearch-plugin list | grep -q analysis-ik
