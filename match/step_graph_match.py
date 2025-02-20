from collections import defaultdict

import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
from occwl.entity_mapper import EntityMapper
import numpy as np
from occwl.io import load_step
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.GeomAbs import *


# 定义节点匹配的函数
def node_match(node1, node2):
    return node2['type'] == 'any' or node1['type'] == node2['type']


# 定义边匹配的函数
def edge_match(edge1, edge2):
    return edge2['type'] == 'any' or edge1['type'] == edge2['type']


# 检查数值相等
def is_equal(value1, value2, tol=1e-4):
    return value1 - value2 < tol


# 检查点重合
def is_coincide(point1, point2, tol=1e-4):
    return is_equal(point1[0], point2[0], tol) and is_equal(point1[1], point2[1], tol) and is_equal(point1[2],
                                                                                                    point2[2], tol)


# 检查方向平行
def is_parallel(axis1, axis2, tol=1e-4):
    cross_product = np.cross(axis1, axis2)
    return np.linalg.norm(cross_product) < tol


def is_perpendicular(axis1, axis2, tol=1e-4):
    return np.dot(axis1, axis2) < tol


# 检查轴共线
def is_collinear(axis1, axis2, point1, point2, tol=1e-4):
    axis3 = [
        point2[0] - point1[0],
        point2[1] - point1[1],
        point2[2] - point1[2]
    ]
    return is_parallel(axis1, axis2, tol) and is_parallel(axis1, axis3, tol)


class StepGraphMatcher(GraphMatcher):
    def __init__(self, G1, G2, node_match=None, edge_match=None, comparisons=None):
        super().__init__(G1, G2, node_match, edge_match)
        if comparisons is None:
            comparisons = []
        self.nodes_num = len(self.G2)
        self.comparisons = comparisons

    def semantic_feasibility(self, G1_node, G2_node):
        # 调用默认的节点匹配逻辑
        if not super().semantic_feasibility(G1_node, G2_node):
            return False
        if len(self.mapping) < self.nodes_num:
            return True

        rmapping = {v: k for k, v in self.mapping.items()}

        for comp in self.comparisons:
            a1, a2 = self.get_comp_obj(comp['a'], rmapping)
            b1, b2 = self.get_comp_obj(comp['b'], rmapping)
            if not self.compare(comp['func'], a1, b1, a2, b2):
                return False
        return True

    def get_comp_obj(self, a, mapping):
        n1 = mapping[a['n1']]
        if 'n2' in a:
            n2 = mapping[a['n2']]
            foe = self.G1[n1][n2]
        else:
            foe = self.G1.nodes[n1]
        if 'param2' in a:
            return foe[a['param1']], foe[a['param2']]
        else:
            return foe[a['param1']], None

    def compare(self, func, a, b, c, d):
        if func == 'eq':
            return is_equal(a, b)
        if func == 'ne':
            return not is_equal(a, b)
        if func == 'gt':
            return a > b
        if func == 'lt':
            return a < b
        if func == 'coincide':
            return is_coincide(a, b)
        if func == 'parallel':
            return is_parallel(a, b)
        if func == 'collinear':
            return is_collinear(a, b, c, d)


def cal_param(p1, p2, func):
    if func == "dist":
        return np.linalg.norm(np.array(p1) - np.array(p2))
    if func == "plus":
        return p1 + p2
    if func == "sub":
        return p1 - p2
    return 0


def get_match_result(graph, subgraph_matches, params_info, parts):
    part_mapper = {}
    for index, part in enumerate(parts):
        for i in part['faces']:
            part_mapper[i] = index

    used_faces = []
    tp_set = set()
    grouped_results = defaultdict(dict)
    for subgraph in subgraph_matches:
        mapper = {v: k for k, v in subgraph.items()}
        params = {}
        params_visible = {}
        tlist = []
        for info in params_info:
            keyword = info["keyword"]
            if info["type"] == "edge":  #边参数
                n1 = info["n1"]
                n2 = info["n2"]
                e = graph[mapper[n1]][mapper[n2]]
                value = e[info["param"]]
            elif info["type"] == "face":  #面参数
                n1 = info["n1"]
                f = graph.nodes[mapper[n1]]
                value = f[info["param"]]
            else:  #计算参数
                p1 = params[info["p1"]]
                p2 = params[info["p2"]]
                value = cal_param(p1, p2, info["func"])
            params[keyword] = value
            if not info['visible']:
                params_visible[keyword] = False
                continue
            params_visible[keyword] = True
            if isinstance(value, list):
                for v in value:
                    tlist.append(round(v, 3))
            else:
                tlist.append(round(value, 3))

        tp = tuple(tlist)

        if tp in tp_set:
            result = grouped_results[tp]
            for i in range(len(mapper)):
                if mapper[i] not in result["faces"] and i in part_mapper:
                    part_index = part_mapper[i]
                    result["faces"].append(mapper[i])
                    result["parts"][part_index]["faces"].append(mapper[i])
        else:
            result = {
                "faces": [v for k, v in mapper.items()],
                "param": [
                    {
                        "name": k,
                        "value": v
                    } for k, v in params.items() if params_visible[k]
                ],
                "parts": []
            }
            for part in parts:
                part_name = part["name"]
                part_faces = [mapper[i] for i in part["faces"]]
                result["parts"].append({
                    "name": part_name,
                    "faces": part_faces
                })
            grouped_results[tp] = result
            tp_set.add(tp)

    retlist = []

    # 避免同一个面属于多个相同结构
    for k, values in grouped_results.items():
        faces = values["faces"]
        if any(face in faces for face in used_faces):
            print(f"重复结果: {faces}")
        else:
            used_faces += faces
            values["faces"] = []
            for part in values["parts"]:
                values["faces"].append(part["faces"])
            retlist.append(values)

    return retlist


def match(solid, matcher):
    mapper = EntityMapper(solid)
    graph = nx.Graph()
    for face in solid.faces():
        face_idx = mapper.face_index(face)
        graph.add_node(face_idx)
        surf = BRepAdaptor_Surface(face.topods_shape())
        surf_type = surf.GetType()
        if surf_type == GeomAbs_Plane:
            gp_pln = surf.Plane()
            normal = gp_pln.Axis().Direction()
            graph.nodes[face_idx]['type'] = "plain"
            graph.nodes[face_idx]['normal'] = [normal.X(), normal.Y(), normal.Z()]
        elif surf_type == GeomAbs_Cylinder:
            gp_cyl = surf.Cylinder()
            location = gp_cyl.Location()
            direction = gp_cyl.Axis().Direction()
            radius = gp_cyl.Radius()
            graph.nodes[face_idx]['type'] = "circle"
            graph.nodes[face_idx]['radius'] = radius
            graph.nodes[face_idx]['direction'] = [direction.X(), direction.Y(), direction.Z()]
            graph.nodes[face_idx]['origin'] = [location.X(), location.Y(), location.Z()]
        elif surf_type == GeomAbs_Cone:
            gp_con = surf.Cone()
            radius = gp_con.RefRadius()
            semi_angle = gp_con.SemiAngle()
            axis = gp_con.Axis()
            location = axis.Location()
            direction = axis.Direction()
            graph.nodes[face_idx]['type'] = "cone"
            graph.nodes[face_idx]['radius'] = radius
            graph.nodes[face_idx]['semiangle'] = semi_angle
            graph.nodes[face_idx]['direction'] = [direction.X(), direction.Y(), direction.Z()]
            graph.nodes[face_idx]['origin'] = [location.X(), location.Y(), location.Z()]
        elif surf_type == GeomAbs_Sphere:
            graph.nodes[face_idx]['type'] = "sphere"
        elif surf_type == GeomAbs_Torus:
            graph.nodes[face_idx]['type'] = "torus"
        else:
            graph.nodes[face_idx]['type'] = "other"

    for edge in solid.edges():
        if not edge.has_curve():
            continue
        connected_faces = list(solid.faces_from_edge(edge))
        if len(connected_faces) == 2:
            left_face, right_face = edge.find_left_and_right_faces(connected_faces)
            if left_face is None or right_face is None:
                continue
            edge_reversed = edge.reversed_edge()
            if not mapper.oriented_edge_exists(edge_reversed):
                continue
            edge_reversed_idx = mapper.oriented_edge_index(edge_reversed)
            left_index = mapper.face_index(left_face)
            right_index = mapper.face_index(right_face)
            graph.add_edge(left_index, right_index)
            curv = BRepAdaptor_Curve(edge.topods_shape())
            curv_type = curv.GetType()
            if curv_type == GeomAbs_Line:
                graph[left_index][right_index]['type'] = "line"
                u_start = curv.FirstParameter()
                u_end = curv.LastParameter()
                start_point = curv.Value(u_start)
                end_point = curv.Value(u_end)
                length = start_point.Distance(end_point)
                graph[left_index][right_index]['length'] = length
            elif curv_type == GeomAbs_Circle:
                gp_circ = curv.Circle()
                location = gp_circ.Location()
                radius = gp_circ.Radius()
                length = gp_circ.Length()
                direction = gp_circ.Axis().Direction()
                graph[left_index][right_index]['type'] = "circle"
                graph[left_index][right_index]['radius'] = radius
                graph[left_index][right_index]['length'] = length
                graph[left_index][right_index]['direction'] = [direction.X(), direction.Y(), direction.Z()]
                graph[left_index][right_index]['origin'] = [location.X(), location.Y(), location.Z()]
            else:
                graph[left_index][right_index]['type'] = "other"

    template = nx.Graph()
    template.add_nodes_from(matcher["nodes"])
    template.add_edges_from(matcher["edges"])

    graph_matcher = StepGraphMatcher(graph, template, node_match=node_match, edge_match=edge_match,
                                     comparisons=matcher["comparisons"])

    subgraph_matches = graph_matcher.subgraph_isomorphisms_iter()
    result = get_match_result(graph, subgraph_matches, matcher["params"], matcher["parts"])
    return result




def get_matchers():  #这里正常是通过查数据库获取，改为直接定义两种子图
    matcher1 = {
        'name': '三段阶梯轴',
        'nodes': [
            (0, {"type": "any"}),
            (1, {"type": "circle"}),  # 低圆柱面
            (2, {"type": "plain"}),
            (3, {"type": "circle"}),  # 高圆柱面
            (4, {"type": "plain"}),
            (5, {"type": "circle"}),  # 低圆柱面
            (6, {"type": "any"}),
        ],
        'edges': [
            (0, 1, {'type': 'circle'}),
            (1, 2, {'type': 'circle'}),
            (2, 3, {'type': 'circle'}),
            (3, 4, {'type': 'circle'}),
            (4, 5, {'type': 'circle'}),
            (5, 6, {'type': 'circle'})
        ],
        'comparisons': [
            {
                'a': {
                    'n1': 1,
                    'param1': 'direction',
                    'param2': 'origin'
                },
                'b': {
                    'n1': 3,
                    'param1': 'direction',
                    'param2': 'origin'
                },
                'func': 'collinear'
            },
            {
                'a': {
                    'n1': 1,
                    'param1': 'direction',
                    'param2': 'origin'
                },
                'b': {
                    'n1': 5,
                    'param1': 'direction',
                    'param2': 'origin'
                },
                'func': 'collinear'
            },
            {
                'a': {
                    'n1': 1,
                    'param1': 'direction',
                    'param2': 'origin'
                },
                'b': {
                    'n1': 3,
                    'param1': 'direction',
                    'param2': 'origin'
                },
                'func': 'collinear'
            },
            {
                'a': {
                    'n1': 1,
                    'param1': 'radius'
                },
                'b': {
                    'n1': 3,
                    'param1': 'radius'
                },
                'func': 'lt'
            },

        ],
        'params': [
            {
                'keyword': '轴段1-半径',
                'type': 'face',
                'n1': 1,
                'param': 'radius',
                'visible': True
            },
            {
                'keyword': '轴段2-半径',
                'type': 'face',
                'n1': 3,
                'param': 'radius',
                'visible': True
            },
            {
                'keyword': '轴段3-半径',
                'type': 'face',
                'n1': 5,
                'param': 'radius',
                'visible': True
            },
            {
                'keyword': '圆心01',
                'type': 'edge',
                'n1': 0,
                'n2': 1,
                'param': 'origin',
                'visible': False
            },
            {
                'keyword': '圆心12',
                'type': 'edge',
                'n1': 1,
                'n2': 2,
                'param': 'origin',
                'visible': False
            },
            {
                'keyword': '圆心23',
                'type': 'edge',
                'n1': 2,
                'n2': 3,
                'param': 'origin',
                'visible': False
            },
            {
                'keyword': '圆心34',
                'type': 'edge',
                'n1': 3,
                'n2': 4,
                'param': 'origin',
                'visible': False
            },
            {
                'keyword': '圆心45',
                'type': 'edge',
                'n1': 4,
                'n2': 5,
                'param': 'origin',
                'visible': False
            },
            {
                'keyword': '圆心56',
                'type': 'edge',
                'n1': 5,
                'n2': 6,
                'param': 'origin',
                'visible': False
            },
            {
                'keyword': '轴段1-长度',
                'type': 'cal',
                'p1': '圆心01',
                'p2': '圆心12',
                'func': 'dist',
                'visible': True
            },
            {
                'keyword': '轴段2-长度',
                'type': 'cal',
                'p1': '圆心23',
                'p2': '圆心34',
                'func': 'dist',
                'visible': True
            },
            {
                'keyword': '轴段3-长度',
                'type': 'cal',
                'p1': '圆心45',
                'p2': '圆心56',
                'func': 'dist',
                'visible': True
            },
            {
                'keyword': '轴段1-深度',
                'type': 'cal',
                'p1': '轴段2-半径',
                'p2': '轴段1-半径',
                'func': 'sub',
                'visible': True
            },
            {
                'keyword': '轴段3-深度',
                'type': 'cal',
                'p1': '轴段2-半径',
                'p2': '轴段3-半径',
                'func': 'sub',
                'visible': True
            },
        ],
        'parts': [
            {
                'name': '轴段1',
                'faces': [1]
            },
            {
                'name': '轴段2',
                'faces': [2, 3, 4]
            },
            {
                'name': '轴段3',
                'faces': [5]
            },
        ]
    }
    matcher2 = {
        'name': '凹槽',
        'nodes': [
            (0, {"type": "any"}),  # 凹槽外面
            (1, {"type": "circle"}),  # 左圆弧柱面
            (2, {"type": "plain"}),  # 竖直面
            (3, {"type": "circle"}),  # 右圆弧柱面
            (4, {"type": "plain"})  # 底面
        ],
        'edges': [
            (0, 1, {'type': 'any'}),
            (0, 2, {'type': 'line'}),
            (0, 3, {'type': 'any'}),
            (1, 2, {'type': 'line'}),
            (1, 4, {'type': 'circle'}),
            (2, 3, {'type': 'line'}),
            (2, 4, {'type': 'line'}),
            (3, 4, {'type': 'circle'}),
        ],
        'comparisons': [
            {
                'a': {
                    'n1': 1,
                    'param1': 'direction'
                },
                'b': {
                    'n1': 3,
                    'param1': 'direction'
                },
                'func': 'parallel'
            },
            {
                'a': {
                    'n1': 1,
                    'param1': 'direction'
                },
                'b': {
                    'n1': 4,
                    'param1': 'normal'
                },
                'func': 'parallel'
            },
            {
                'a': {
                    'n1': 1,
                    'param1': 'radius'
                },
                'b': {
                    'n1': 3,
                    'param1': 'radius'
                },
                'func': 'eq'
            }
        ],
        'params': [
            {
                'keyword': '凹槽深度',
                'type': 'edge',
                'n1': 1,
                'n2': 2,
                'param': 'length',
                'visible': True
            },
            {
                'keyword': '矩形长度',
                'type': 'edge',
                'n1': 2,
                'n2': 4,
                'param': 'length',
                'visible': True
            },
            {
                'keyword': '弧面半径',
                'type': 'face',
                'n1': 1,
                'param': 'radius',
                'visible': True
            },
            {
                'keyword': '圆心1',
                'type': 'edge',
                'n1': 1,
                'n2': 4,
                'param': 'origin',
                'visible': False
            },
            {
                'keyword': '圆心2',
                'type': 'edge',
                'n1': 3,
                'n2': 4,
                'param': 'origin',
                'visible': False
            },
            {
                'keyword': '矩形宽度',
                'type': 'cal',
                'p1': '弧面半径',
                'p2': '弧面半径',
                'func': 'plus',
                'visible': True
            },
        ],
        'parts': [
            {
                'name': '凹槽底面',
                'faces': [4]
            },
            {
                'name': '圆弧柱面',
                'faces': [1, 3]
            },
            {
                'name': '竖直面',
                'faces': [2]
            },
        ]
    }
    mathers = [matcher1, matcher2]
    return mathers

def graph_match(fn):
    solids = load_step(fn)
    assert len(solids) == 1, "step_parse error"
    solid = load_step(fn)[0]
    mathers = get_matchers()
    feathers = []
    for mather in mathers:
        feather = match(solid, mather)
        if len(feather) > 0:
            feathers.append({
                "name": mather['name'],
                "list": feather
            })
    return feathers