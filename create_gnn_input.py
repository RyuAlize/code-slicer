import csv
import numpy as np
import functools
from util import read_csv

block_root = ['ReturnStatement', 'DoStatement', 'GotoStatement']


type_map = {
    'AndExpression': 1, 'Sizeof': 2, 'Identifier': 3, 'ForInit': 4, 'ReturnStatement': 5, 'SizeofOperand': 6,
    'InclusiveOrExpression': 7, 'PtrMemberAccess': 8, 'AssignmentExpression': 9, 'ParameterList': 10,
    'IdentifierDeclType': 11, 'SizeofExpression': 12, 'SwitchStatement': 13, 'IncDec': 14, 'Function': 15,
    'BitAndExpression': 16, 'UnaryExpression': 17, 'DoStatement': 18, 'GotoStatement': 19, 'Callee': 20,
    'OrExpression': 21, 'ShiftExpression': 22, 'Decl': 23, 'CFGErrorNode': 24, 'WhileStatement': 25,
    'InfiniteForNode': 26, 'RelationalExpression': 27, 'CFGExitNode': 28, 'Condition': 29, 'BreakStatement': 30,
    'CompoundStatement': 31, 'UnaryOperator': 32, 'CallExpression': 33, 'CastExpression': 34,
    'ConditionalExpression': 35, 'ArrayIndexing': 36, 'PostIncDecOperationExpression': 37, 'Label': 38,
    'ArgumentList': 39, 'EqualityExpression': 40, 'ReturnType': 41, 'Parameter': 42, 'Argument': 43, 'Symbol': 44,
    'ParameterType': 45, 'Statement': 46, 'AdditiveExpression': 47, 'PrimaryExpression': 48, 'DeclStmt': 49,
    'CastTarget': 50, 'IdentifierDeclStatement': 51, 'IdentifierDecl': 52, 'CFGEntryNode': 53, 'TryStatement': 54,
    'Expression': 55, 'ExclusiveOrExpression': 56, 'ClassDef': 57, 'File': 58, 'UnaryOperationExpression': 59,
    'ClassDefStatement': 60, 'FunctionDef': 61, 'IfStatement': 62, 'MultiplicativeExpression': 63,
    'ContinueStatement': 64, 'MemberAccess': 65, 'ExpressionStatement': 66, 'ForStatement': 67, 'InitializerList': 68,
    'ElseStatement': 69
}

type_one_hot = np.eye(len(type_map))

def compare(a,b):
    return int(a) < int(b)

def checkHex(s: str) -> bool:  
    if s.startswith('0x') or s.startswith('0X'):
        s = s[2:] 
        for c in s:
            if not ( (c >= '0' and c <='9') or (c >='a' and c <= 'f') or (c >= 'A' and c <= 'F')):
                return False
        return True
    elif s.startswith('~0X') or s.startswith('~0x'):
        s = s[3:]
        for c in s:
            if not ( (c >= '0' and c <='9') or (c >='a' and c <= 'f') or (c >= 'A' and c <= 'F')):
                return False
        return True
    return False


def is_numic(token: str) -> bool :
    if token.isdigit():
        return True
    if checkHex(token):
        return True
    return False
    
def is_str_litrial(token: str) -> bool:
    if token.startswith('"') and token.endswith('"'):
        return True
    if token.startswith("'") and token.endswith("'"):
        return True
    return False
def split_tokens(doc):
    tokens = doc.split(' ')
    words = []
    for token in tokens:
        if token == 'L':
            continue
        elif is_numic(token):
            words.append('NUM')
        elif is_str_litrial(token):
            words.append('STRING')
        else:
            words.append(token)
    return words


def inputGeneration(filter_node_ids, nodeCSV, edgeCSV, target, wv, edge_type_map:dict, cfg_only=False):
    gInput = dict()
    gInput["targets"] = list()
    gInput["graph"] = list()
    gInput["node_features"] = list()
    gInput["targets"].append([target])


    with open(nodeCSV, 'r') as nc:
        nodes = csv.DictReader(nc, delimiter='\t')
        nodeMap = dict()
        allNodes = {}
        node_idx = 0
        for idx, node in enumerate(nodes):
            if node['key'] not in filter_node_ids:
                continue
            nodeKey = node['key']
            node_type = node['type']
            if node_type == 'File':
                continue
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
            node_feature = type_one_hot[type_map[node_type] - 1].tolist() 
            node_feature.extend(fNrp.tolist())
            allNodes[nodeKey] = node_feature
            nodeMap[nodeKey] = node_idx
            node_idx += 1
        if node_idx == 0 or node_idx >= 500:
            return None
        all_nodes_with_edges = set()
        trueNodeMap = {}
        all_edges = []
        with open(edgeCSV, 'r') as ec:
            reader = csv.DictReader(ec, delimiter='\t')
            for e in reader:
                start, end, eType = e["start"], e["end"], e["type"]
                if (start not in filter_node_ids or end not in filter_node_ids) or eType not in edge_type_map.keys():
                    continue
                
                all_nodes_with_edges.add(start)
                all_nodes_with_edges.add(end)
                edge = [start, edge_type_map[eType], end]
                all_edges.append(edge)
        if len(all_edges) == 0:
            return None
        for i, node in enumerate(all_nodes_with_edges):
            trueNodeMap[node] = i
            gInput["node_features"].append(allNodes[node])
        for edge in all_edges:
            start, t, end = edge
            start = trueNodeMap[start]
            end = trueNodeMap[end]
            e = [start, t, end]
            gInput["graph"].append(e)
    return gInput



def inputGeneration2(slice_line_keys,filter_node_keys, nodeCSV, edgeCSV, target, wv, edge_type_map:dict, cfg_only=False):
    gInput = dict()
    gInput["targets"] = list()
    gInput["graph"] = list()
    gInput["node_features"] = list()
    gInput["targets"].append([target])

    nodes = read_csv(nodeCSV)
    edges = read_csv(edgeCSV)

    all_nodes_with_edges = set()
    trueNodeMap = {}
    all_edges = []
    allNodes = {}

    for e in edges:
        start, end, eType = e["start"], e["end"], e["type"]
        if (start not in filter_node_keys or end not in filter_node_keys) or eType not in edge_type_map.keys():
            continue           
        all_nodes_with_edges.add(start)
        all_nodes_with_edges.add(end)
        edge = [start, edge_type_map[eType], end]
        all_edges.append(edge)
    
    if len(all_nodes_with_edges) == 0 or len(all_nodes_with_edges) > 500:
        return None

    all_node_keys =  list(all_nodes_with_edges)
    all_node_keys.sort(key = lambda a: int(a))
    slice_line_keys.sort(key = lambda a: int(a))

    for idx in range(len(all_node_keys)):
        trueNodeMap[all_node_keys[idx]] = idx

    for node in nodes:
        if node['key'] not in all_nodes_with_edges:
            continue
        nodeKey = node['key'] 
        node_type = node['type']
        if node_type == 'File':
            continue
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
        node_feature = type_one_hot[type_map[node_type] - 1].tolist() 
        node_feature.extend(fNrp.tolist())
        
        allNodes[trueNodeMap[nodeKey]] = node_feature

    for i in range(len(all_node_keys)):
        gInput["node_features"].append(allNodes[i])

    for edge in all_edges:
        start, t, end = edge
        start = trueNodeMap[start]
        end = trueNodeMap[end]
        e = [start, t, end]
        gInput["graph"].append(e)
    
    t = len(edge_type_map)+1
    for i in range(1,len(slice_line_keys)):
        e = [trueNodeMap[slice_line_keys[i-1]], t, trueNodeMap[slice_line_keys[i]]]
        gInput["graph"].append(e)
    return gInput


