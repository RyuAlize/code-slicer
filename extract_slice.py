import csv
import numpy as np
import os
import re 
import os
import json
from util import extract_statement_node, get_ast, read_csv, read_code_file, extract_nodes_with_location_info, create_adjacency_list,combine_control_and_data_adjacents,create_forward_slice,create_backward_slice
from helper import l_funcs

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

def get_slice_data(slice_lines, combined_graph, adjacency_list):
    _keys = set()
    slice_datas = []
    for slice_ln in slice_lines:
        forward_sliced_lines = create_forward_slice(combined_graph, slice_ln)
        backward_sliced_lines = create_backward_slice(combined_graph, slice_ln)
        all_slice_lines = forward_sliced_lines
        all_slice_lines.extend(backward_sliced_lines)
        all_slice_lines = sorted(list(set(all_slice_lines)))
        key = ' '.join([str(i) for i in all_slice_lines])
        if key not in _keys:
            cdg = []
            ddg = []
            order = []

            for i in range(1,len(all_slice_lines)):
                order.append([all_slice_lines[i-1], all_slice_lines[i]])

            for src_ln, target_ln_set in adjacency_list.items():
                if src_ln not in all_slice_lines:
                    continue
                for ln in target_ln_set[0]:
                    if ln in all_slice_lines and ln != src_ln:
                        cdg.append([src_ln, ln])
                for ln in target_ln_set[1]:
                    if ln in all_slice_lines and ln != src_ln:
                        ddg.append([src_ln, ln])
            edges = {
                'cdg': cdg,
                'ddg':ddg,
                'order':order,
            }
            slice_data = {}
            slice_data['edges'] = edges
            slice_data['ln'] = all_slice_lines
            slice_datas.append(slice_data)

    return slice_datas

def get_slice_nodes_set(nodes):

    call_nodes = set()
    array_nodes = set()
    ptr_nodes = set()
    arithmatic_nodes = set()

    for node_idx, node in enumerate(nodes):
        ntype = node['type'].strip()
        if ntype == 'CallExpression':
            function_name = nodes[node_idx + 1]['code']
            if function_name  is None or function_name.strip() == '':
                continue
            if function_name.strip() in l_funcs:
                call_nodes.add(node['key'])
        elif ntype == 'ArrayIndexing':
            array_nodes.add(node['key'])
        elif ntype == 'PtrMemberAccess':
            ptr_nodes.add(node['key'])
        elif node['operator'].strip() in ['+', '-', '*', '/']:
            arithmatic_nodes.add(node['key'])
    
            
    slice_nodes_set = {     
        'call_nodes': call_nodes,    
        'array_nodes': array_nodes,     
        'arith_nodes': arithmatic_nodes,      
        'ptr_nodes': ptr_nodes,
    }
     
    return slice_nodes_set

def get_slice(nodes, edges, file_path):

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
    
    node_indices, node_ids, line_numbers, node_id_to_ln = extract_nodes_with_location_info(nodes)
    adjacency_list = create_adjacency_list(line_numbers, node_id_to_ln, edges, False)
    combined_graph = combine_control_and_data_adjacents(adjacency_list)

    array_slices = get_slice_data(array_lines, combined_graph, adjacency_list)
    call_slices = get_slice_data(call_lines, combined_graph, adjacency_list)
    arith_slices = get_slice_data(arithmatic_lines, combined_graph, adjacency_list)
    ptr_slices = get_slice_data(ptr_lines, combined_graph, adjacency_list)
            
    data_instance = {
        'file_path': file_path,     
        'call_slices': call_slices,    
        'array_slices_vd': array_slices,     
        'arith_slices_vd': arith_slices,      
        'ptr_slices_vd': ptr_slices,
    }
     
    return data_instance


def get_statement_gtaph(nodes, edges, file_path):
    control_edges, data_edges = list(), list()
    statement_nodes, key_to_nodes = extract_statement_node(nodes)
    _,ast_unionset = get_ast(edges, key_to_nodes)
    statement_set = set()
    for edge in edges:
        edge_type = edge['type'].strip()
        if True:  # edge_type in ['IS_AST_PARENT', 'FLOWS_TO']:
            start_node_id = edge['start'].strip()
            while ast_unionset.get(start_node_id) is not None and not key_to_nodes[start_node_id]['type'].endswith('Statement'):
                start_node_id = ast_unionset.get(start_node_id)
            end_node_id = edge['end'].strip()
            while ast_unionset.get(end_node_id) is not None and not key_to_nodes[start_node_id]['type'].endswith('Statement'):
                end_node_id = ast_unionset.get(end_node_id)
            if start_node_id not in statement_nodes or end_node_id not in statement_nodes:
                continue
            if edge_type == 'CONTROLS' and (start_node_id, end_node_id) not in control_edges:  # Control
                control_edges.append((start_node_id, end_node_id))
                statement_set.add(start_node_id)
                statement_set.add(end_node_id)
            if edge_type == 'REACHES' and (start_node_id, end_node_id) not in data_edges:  # Data
                data_edges.append((start_node_id, end_node_id))
                statement_set.add(start_node_id)
                statement_set.add(end_node_id)
    
    sorte_stm= list(statement_set)
    sorte_stm.sort(key=lambda a: int(a))
    index_key_map = {}
    for i, key in enumerate(sorte_stm) :
        index_key_map[key] = i
    order_edges = []
    for i in range(1, len(sorte_stm)):
        order_edges.append([index_key_map[sorte_stm[i-1]], index_key_map[sorte_stm[i]]])
    
    new_data_edges = [[index_key_map[e[0]], index_key_map[e[1]]] for e in  data_edges]
    new_control_edges = [[index_key_map[e[0]], index_key_map[e[1]]] for e in  control_edges]
    code = [key_to_nodes[key]['code'] for key in sorte_stm]
    
    pdg = {
        'code': code,
        'ddg': new_data_edges,
        'cdg': new_control_edges,
        'order': order_edges,
    }
    return pdg