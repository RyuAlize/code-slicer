import os
import numpy as np
import json
from util import extract_nodes_with_location_info, read_code_file,read_csv
from create_gnn_input import type_map,type_one_hot
from code_process import parse_token


base = "/home/zhangxs/data/ReVeal-master/code-slicer/upload_source_1"

def joern_parse(dir_name, file_name):
    os.system('rm -r  tmp' )
    os.system('mkdir -p tmp')
    os.system('rm -r  parsed' )
    file_path = os.path.join(dir_name, file_name)
    os.system('cp ' + file_path + ' tmp/'+file_name)
    os.system('./joern/joern-parse tmp/')

def split_tokens(code):
    tokens = []
    for token in code.split(' '):
            parse_token(token, tokens)
    
    return tokens 

def ast_edge_process(nodes, edges, target, wv):
    gInput = dict()
    gInput["targets"] = list()
    gInput["graph"] = list()
    gInput["node_features"] = list()
    gInput["targets"].append([target])

    node_key_map = {}
    for node in nodes:
        node_key_map[node['key']] = node

    ast_node_keys =set()
    key_index_map = {}
    index_key_map = {}
    for edge in edges:       
        ast_node_keys.add(edge['start'])
        ast_node_keys.add(edge['end'])

    ast_nodes = [node  for node in nodes if node['key'] in ast_node_keys]

    for i, node in enumerate(ast_nodes):
        key_index_map[i] = node['key']
        index_key_map[node['key']] = i 
        node_content = node['code'].strip()
        node_split = split_tokens(node_content)
        nrp = np.zeros(50)
        for token in node_split:
            try: 
                embedding = wv.wv[token]
            except:
                embedding = np.zeros(50)
            nrp = np.add(nrp, embedding)
        if len(node_split) > 0:
            fNrp = np.divide(nrp, len(node_split))
        else:
            fNrp = nrp
        node_feature = type_one_hot[type_map[node['type']] - 1].tolist() 
        node_feature.extend(fNrp.tolist())
        gInput["node_features"].append(node_feature)   

    for edge in edges:
        start_key = edge['start']
        end_key = edge['end']
        gInput["graph"].append([index_key_map[start_key], 1, index_key_map[end_key]])
    
    return gInput

def get_pdg_graph(nodes, edges):
    control_edges, data_edges = list(), list()
    node_indices, node_ids, line_numbers, node_id_to_ln = extract_nodes_with_location_info(nodes)
    for edge in edges:
        edge_type = edge['type'].strip()
        if True:  # edge_type in ['IS_AST_PARENT', 'FLOWS_TO']:
            start_node_id = edge['start'].strip()
            end_node_id = edge['end'].strip()
            if start_node_id not in node_id_to_ln.keys() or end_node_id not in node_id_to_ln.keys():
                continue
            start_ln = node_id_to_ln[start_node_id]
            end_ln = node_id_to_ln[end_node_id]
            if edge_type == 'CONTROLS':  # Control
                control_edges.append((start_ln, end_ln))
            if edge_type == 'REACHES':  # Data
                data_edges.append((start_ln, end_ln))

    pdg = {
        'ddg': data_edges,
        'cdg': control_edges,
    }
    return pdg

def graph_process(dir_name, file, label, wv=None):
    joern_parse(dir_name, file)
    nodes_path = os.path.join('parsed', 'tmp',file,'nodes.csv')
    edges_path = os.path.join('parsed', 'tmp', file, 'edges.csv')
    nodes = read_csv(nodes_path)
    edges = read_csv(edges_path)

    pdg = get_pdg_graph(nodes, edges)   

    ast_graph = {
        'file_name':file,
        'graph':pdg,
        'label': label,
    }
    return ast_graph


def statement_graph(line:str)-> dict:
    tmp = line.split(' ')
    file_path, split_line, label = tmp[1], tmp[2].strip(), tmp[3].strip()

    graph_process(file_path, )






labelf = open('/home/zhangxs/data/ReVeal-master/data/codexglue/label.json','r')
labels = json.load(labelf)
label = {}
for data in labels:
    label[data['file_name']] = data['label']
ast_graph('/home/zhangxs/data/ReVeal-master/data/codexglue/train', 'dataset/codexglue-train-pdg.json', label)
#ast_graph_process('/home/zhangxs/data/ReVeal-master/data/codexglue/train', 'FFmpeg00a1e1337f22376909338a5319a378b2e2afdde8.c')