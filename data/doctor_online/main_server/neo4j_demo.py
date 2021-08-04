# 导入一个包
from neo4j import GraphDatabase
# 导入neo4j数据库的配置信息，包含图数据库的URL地址，端口号，用户名，密码
from config import NEO4J_CONFIG

# 创建一个neo4j驱动对象
driver = GraphDatabase.driver(**NEO4J_CONFIG)
# 所有的命令操作都需要先创建会话
'''with driver.session() as session:
    cypher = "CREATE (c:Company) SET c.name='🐮' RETURN c.name"
    record = session.run(cypher)
    result = list(map(lambda x: x[0], record))
    print("result:", result)'''
# 创建一个事务函数
def _some_operations(tx, cat_name, mouse_name):
    tx.run("MERGE (a:Cat{name: $cat_name})"
           "MERGE (b:Mouse{name: $mouse_name})"
           "MERGE (a)-[r:And]-(b)", cat_name=cat_name, mouse_name=mouse_name)

# 所有的事务都要开启会话
with driver.session() as session:
    session.write_transaction(_some_operations, "TOM", "Jery")
