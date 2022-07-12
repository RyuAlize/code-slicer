import csv
import os
from graphviz import Digraph
import networkx as nx
from typing import List, Set, Tuple, Dict

import scipy as sp
from helper import l_funcs

def get_vul_line(file_path):
    vul_lines = {}
    with open(file_path, 'r') as f:
        i = 1
        for line in f.readlines():
            if "//FAULTY_F" in line:
                line = line[:line.find("//FAULTY_F")]
                vul_lines[i]= line.strip(' ')
    return vul_lines

def joern_parse_file(file_path: str):
    file = file_path.split('/')[-1]
    os.system('rm -r  tmp' )
    os.system('mkdir -p tmp')
    os.system('rm -r  parsed' )
    os.system('cp ' + file_path + ' tmp/'+file)
    os.system('./joern/joern-parse tmp/')
    return file

def joern_parse(dir_name: str, files: List[str]):
    os.system('rm -r  tmp' )
    os.system('mkdir -p tmp')
    os.system('rm -r  parsed' )
    for file in files:
        file_path = os.path.join(dir_name, file)
        os.system('cp ' + file_path + ' tmp/'+file)
    os.system('./joern/joern-parse tmp/')

def read_parsed(file) -> Tuple[List[Dict], List[Dict]]:
    nodes_path = os.path.join('parsed', 'tmp',file,'nodes.csv')
    edges_path = os.path.join('parsed', 'tmp', file, 'edges.csv')
    nodes = read_csv(nodes_path)
    edges = read_csv(edges_path)
    return nodes, edges

def extract_line_number(idx, nodes):
    while idx >= 0:
        c_node = nodes[idx]
        if 'location' in c_node.keys():
            location = c_node['location']
            if location.strip() != '':
                try:
                    ln = int(location.split(':')[0])
                    return ln
                except:
                    pass
        idx -= 1
    return -1

def get_slice_criteria(nodes: List[Dict]) -> Dict :
    call_lines = set()
    array_lines = set()
    ptr_lines = set()
    arithmatic_lines = set()

    for node_idx, node in enumerate(nodes):
        ntype = node['type'].strip()
        if ntype == 'CallExpression':
            function_name = nodes[node_idx + 1]['code']
            if function_name  is None or function_name.strip() == '':
                continue
            if function_name.strip() in l_funcs:
                line_no = extract_line_number(node_idx, nodes)
                if line_no > 0:
                    call_lines.add(line_no)
        elif ntype == 'ArrayIndexing':
            line_no = extract_line_number(node_idx, nodes)
            if line_no > 0:
                array_lines.add(line_no)
        elif ntype == 'PtrMemberAccess':
            line_no = extract_line_number(node_idx, nodes)
            if line_no > 0:
                ptr_lines.add(line_no)
        elif node['operator'].strip() in ['+', '-', '*', '/']:
            line_no = extract_line_number(node_idx, nodes)
            if line_no > 0:
                arithmatic_lines.add(line_no)
    
    slice_criteria = {
        'call_lines': call_lines,
        'array_lines':array_lines,
        'ptr_lines':ptr_lines,
        'arithmatic_lines':arithmatic_lines
    }
    return slice_criteria

def read_csv(csv_file_path):
    data = []
    with open(csv_file_path) as fp:
        header = fp.readline()
        header = header.strip()
        h_parts = [hp.strip() for hp in header.split('\t')]
        for line in fp:
            line = line.strip()
            instance = {}
            lparts = line.split('\t')
            for i, hp in enumerate(h_parts):
                if i < len(lparts):
                    content = lparts[i].strip()
                else:
                    content = ''
                instance[hp] = content
            data.append(instance)
        return data


def read_code_file(file_path):
    code_lines = {}
    with open(file_path) as fp:
        for ln, line in enumerate(fp):
            assert isinstance(line, str)
            line = line.strip()
            if '//' in line:
                line = line[:line.index('//')]
            if '/*' in line:
                line = line[:line.index('/*')]
            code_lines[ln + 1] = line
        return code_lines


def extract_statement_node(nodes):
    statement_nodes = []
    key_to_nodes = {}
    for node in nodes:
        assert isinstance(node, dict)
        key_to_nodes[node['key'].strip()] = node
        if node['type'].endswith('Statement'):
            node_id = node['key'].strip()
            statement_nodes.append(node_id)
    return statement_nodes, key_to_nodes

def extract_nodes_with_location_info(nodes):
    node_key_map = {}
    node_ids = []
    line_numbers = []
    node_id_to_line_number = {}
    line_number_to_node_id = {}
    for node_index, node in enumerate(nodes):
        node_key_map[node['key']] = node
        assert isinstance(node, dict)
        if 'location' in node.keys() and node['type'].endswith('Statement'):
            location = node['location']
            if location == '':
                continue
            line_num = int(location.split(':')[0])
            node_id = node['key'].strip()
            node_ids.append(node_id)
            line_numbers.append(line_num)
            node_id_to_line_number[node_id] = line_num
            line_number_to_node_id[line_num] = node_id
    return node_key_map, node_ids, line_numbers, node_id_to_line_number, line_number_to_node_id


def create_adjacency_list(line_numbers, node_id_to_line_numbers, edges, data_dependency_only=False):
    adjacency_list = {}
    for ln in set(line_numbers):
        adjacency_list[ln] = [set(), set()]
    for edge in edges:
        edge_type = edge['type'].strip()
        if True :#edge_type in ['IS_AST_PARENT', 'FLOWS_TO']:
            start_node_id = edge['start'].strip()
            end_node_id = edge['end'].strip()
            if start_node_id not in node_id_to_line_numbers.keys() or end_node_id not in node_id_to_line_numbers.keys():
                continue
            start_ln = node_id_to_line_numbers[start_node_id]
            end_ln = node_id_to_line_numbers[end_node_id]
            if not data_dependency_only:
                if edge_type == 'CONTROLS': #Control Flow edges
                    adjacency_list[start_ln][0].add(end_ln)
            if edge_type == 'REACHES': # Data Flow edges
                adjacency_list[start_ln][1].add(end_ln)
    return adjacency_list



def create_visual_graph(code, adjacency_list, file_name='/home/zhangxs/data/ReVeal-master/code-slicer/test_graph', verbose=False):
    graph = Digraph('Code Property Graph')
    for ln in adjacency_list:
        graph.node(str(ln), str(ln) + '\t' + code[ln], shape='box')
        control_dependency, data_dependency = adjacency_list[ln]
        for anode in control_dependency:
            graph.edge(str(ln), str(anode), color='red')
        for anode in data_dependency:
            graph.edge(str(ln), str(anode), color='blue')
    graph.render(file_name, view=verbose, format='png')
    print(graph)


def create_forward_slice(adjacency_list, line_no):
    sliced_lines = set()
    sliced_lines.add(line_no)
    stack = list()
    stack.append(line_no)
    while len(stack) != 0:
        cur = stack.pop()
        if cur not in sliced_lines:
            sliced_lines.add(cur)
        adjacents = adjacency_list[cur]
        for node in adjacents:
            if node not in sliced_lines:
                stack.append(node)
    sliced_lines = sorted(sliced_lines)
    return sliced_lines


def combine_control_and_data_adjacents(adjacency_list):
    cgraph = {}
    for ln in adjacency_list:
        cgraph[ln] = set()
        cgraph[ln] = cgraph[ln].union(adjacency_list[ln][0])
        cgraph[ln] = cgraph[ln].union(adjacency_list[ln][1])
    return cgraph


def invert_graph(adjacency_list):
    igraph = {}
    for ln in adjacency_list.keys():
        igraph[ln] = set()
    for ln in adjacency_list:
        adj = adjacency_list[ln]
        for node in adj:
            igraph[node].add(ln)
    return igraph
    pass

def create_backward_slice(adjacency_list, line_no):
    inverted_adjacency_list = invert_graph(adjacency_list)
    return create_forward_slice(inverted_adjacency_list, line_no)

def create_block_ast(nodes, edges):
    for edge in edges:
        pass


def build_PDG(node_key_map: Dict, edges: List, node_id_to_ln: Dict) -> nx.DiGraph:
    
    PDG = nx.DiGraph()
    control_edges, data_edges = list(), list()      
 
    for edge in edges:
        edge_type = edge['type'].strip()
        if True:  # edge_type in ['IS_AST_PARENT', 'FLOWS_TO']:
            start_node_id = edge['start'].strip()
            end_node_id = edge['end'].strip()
            if start_node_id not in node_id_to_ln.keys(
            ) or end_node_id not in node_id_to_ln.keys():
                continue
            start_ln = node_id_to_ln[start_node_id]
            end_ln = node_id_to_ln[end_node_id]
            if start_ln == end_ln:
                continue
            if node_key_map[end_node_id]['code'].startswith('break'):
                continue
            if edge_type == 'CONTROLS' :  # Control
                control_edges.append((start_ln, end_ln, {"edge": "c"}))
            if edge_type == 'REACHES':  # Data
                data_edges.append((start_ln, end_ln, {"edge": "d"}))
            if edge_type == 'DOM' and (node_key_map[start_node_id]['type'] != 'Condition') :  # Data 
                if node_key_map[start_node_id]['type'] == 'IdentifierDeclStatement':
                    data_edges.append((start_ln, end_ln, {"edge": "c"}))                 
                elif set(node_key_map[start_node_id]['code'].split(' ')).intersection(set(node_key_map[end_node_id]['code'].split(' '))) is not None:
                    data_edges.append((start_ln, end_ln, {"edge": "c"}))
    PDG.add_edges_from(control_edges)
    PDG.add_edges_from(data_edges)
    return PDG


    
def build_XFG(PDG: nx.DiGraph, split_line:int) -> nx.DiGraph:

    if PDG is None or split_line is None:
        return None
    
    sliced_lines = set()

    # backward traversal
    bqueue = list()
    visited = set()
    bqueue.append(split_line)
    visited.add(split_line)
    while bqueue:
        fro = bqueue.pop(0)
        sliced_lines.add(fro)
        if fro in PDG._pred:
            for pred in PDG._pred[fro]:
                if pred not in visited:
                    visited.add(pred)
                    bqueue.append(pred)

    # forward traversal
    fqueue = list()
    visited = set()
    fqueue.append(split_line)
    visited.add(split_line)
    while fqueue:
        fro = fqueue.pop(0)
        sliced_lines.add(fro)
        if fro in PDG._succ:
            for succ in PDG._succ[fro]:
                if succ not in visited :
                    visited.add(succ)
                    fqueue.append(succ)
    if len(sliced_lines) != 0:
        XFG = PDG.subgraph(list(sliced_lines)).copy()
        XFG.graph["key_line"] = split_line
        
        return XFG
    else:
        return None

def get_arg_line_map(node_key_map: Dict, ln_to_node_id: Dict) -> Dict:
    arg_map = {}   
    for ln, key in ln_to_node_id.items():
        code = node_key_map[key]['code']
        if code.startswith('"') and code.endswith('"'):
                code = code[1:-1]
        tokens: List[str] = code.split(' ')
        if node_key_map[key]['type'] == 'IdentifierDeclStatement':           
            if tokens[-1] == ';':
                tokens.pop()
            arg_map[tokens[-1]] = ln  
    return arg_map

def get_ast(edges:list[Dict], key_to_node: list[Dict]) -> Tuple[Dict,Dict]:
    ast = {}
    unionset = {}
    for edge in edges:
        if edge['type'] == 'IS_AST_PARENT':
            if key_to_node[edge['start']]['type'] not in ['File', 'Function', 'FunctionDef']:
                unionset[edge['end']] = edge['start']
            if ast.get(edge['start']) is None:
                ast[edge['start']] = [edge['end']]
            else:
                ast[edge['start']].append(edge['end'])
    

    return ast, unionset
    
def get_function_call(nodes:list[Dict], fn_name:str) -> Dict:
    for node in nodes:
        if node['type'] == 'ExpressionStatement' and node['code'].startswith(fn_name):
            return node
    return None

def get_function_def(function_node_id: str) -> str:
    return str(int(function_node_id)+1)

def has_arg(node_key_map: dict, func_tion_def_id: str) -> bool:
    function_def_code = node_key_map[func_tion_def_id]['code']
    function_def_code = ''.join(function_def_code.split(' '))
    if function_def_code.endswith('()'):    
        return False
    else:
        return True

def get_all_cfile(dir_name: str) -> List[str]:
    res = []
    for file in os.listdir(dir_name):
        if file.endswith('.c') or file.endswith('.cpp'):
            res.append(file)
    return res

""" vt_map = {}
with open("CDG_list.txt", 'r') as f:
    for line in f.readlines():
        tmp1 = line.split(' ')
        tmp2 = tmp1[1].split('/')
        filename = tmp2[1]
        if filename.startswith('CWE'):
            vt = filename.split('_')[0] 
            if vt_map.get(vt) is None:
                vt_map[vt] = [line]
            else:
                vt_map[vt].append(line)

for vt,lines in vt_map.items():
    with open('cwe/'+vt+'.txt', 'w') as f:
        f.writelines(lines) """

  
