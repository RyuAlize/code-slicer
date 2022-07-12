import os, sys, argparse
from graphviz import Digraph
import pickle
from queue import Queue
from create_gnn_input import inputGeneration, inputGeneration2
import json
from gensim.models.word2vec import Word2Vec
from util import *
from extract_slice import get_slice, get_statement_gtaph, get_slice_nodes_set
from code_process import *

header = ['file_path', 'code', 'label']
datas = []
file_list = "train_filst.txt"
base = "/home/zhangxs/data/ReVeal-master/code-slicer/upload_source_1"

base2 = ""

def create_adjacency_list(line_numbers, node_id_to_line_numbers, edges, data_dependency_only=False):
    adjacency_list = {}
    ast_edges = {}
    for ln in set(line_numbers):
        adjacency_list[ln] = [set(), set()]
    for edge in edges:
        edge_type = edge['type'].strip()
        if edge['type'].strip() == "IS_AST_PARENT":
            start_node_id = edge['start'].strip()
            end_node_id = edge['end'].strip()
            if ast_edges.get(start_node_id) is not None:
                ast_edges[start_node_id].append(end_node_id)
            else:
                ast_edges[start_node_id] = [end_node_id]
        if True :#edge_type in ['IS_AST_PARENT', 'FLOWS_TO']:   
            start_node_id = edge['start'].strip()
            end_node_id = edge['end'].strip()
            if start_node_id not in node_id_to_line_numbers.keys() or end_node_id not in node_id_to_line_numbers.keys():
                continue
            start_ln = node_id_to_line_numbers[start_node_id]
            end_ln = node_id_to_line_numbers[end_node_id]
            if start_ln == end_ln:
                continue
            if not data_dependency_only:
                if edge_type == 'CONTROLS': #Control Flow edges
                    adjacency_list[start_ln][0].add(end_ln)
            if edge_type in ['REACHES','FLOWS_TO','USE']: # Data Flow edges
                adjacency_list[start_ln][1].add(end_ln)
    return adjacency_list, ast_edges


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

def filter_nodes(ast_edges: dict, slice_codeline_nodes:list):
    res_node_ids = []    
    q = Queue()
    for root in slice_codeline_nodes:
        q.put(root)    
        while not q.empty():
            node = q.get()
            if ast_edges.get(node) is not None:
                for child in ast_edges[node]:
                    q.put(child)
            res_node_ids.append(node)
    return res_node_ids

edgeType_control = {
    'IS_AST_PARENT': 1,
    'FLOWS_TO': 2,  # Control Flow
    'CONTROLS': 3,  # Control Dependency edge
}

edgeType_data = {
    'IS_AST_PARENT': 1,
    'DEF': 4,
    'USE': 5,
    'REACHES': 6,
}
edgeType_PDG ={
    'IS_AST_PARENT': 1,
    'FLOWS_TO': 2,
    'CONTROLS': 3,
    'DEF': 4,
    'USE': 5,
    'REACHES': 6,
}
edgeType_full = {
    'IS_AST_PARENT': 1,
    'IS_CLASS_OF': 2,
    'FLOWS_TO': 3,
    'DEF': 4,
    'USE': 5,
    'REACHES': 6,
    'CONTROLS': 7,
    'DECLARES': 8,
    'DOM': 9,
    'POST_DOM': 10,
    'IS_FUNCTION_OF_AST': 11,
    'IS_FUNCTION_OF_CFG': 12
}



def get_slice_graph(line:str, target_path: str):
    tmp = line.split(' ')
    id, file_path, split_line, label = tmp[0], tmp[1].strip(), tmp[-2], tmp[-1].strip()
    
    tmp2 = file_path.split('/')
    dir_name,target_file = tmp2[0], tmp2[1]
    dir_path = os.path.join(base, dir_name)
    all_code_files = get_all_cfile(dir_path)
    joern_parse(dir_path, all_code_files)
  
    parsed_res_map = {}

    for file in all_code_files:
        nodes, edges = read_parsed(file)
        node_key_map, node_ids, line_numbers, node_id_to_ln, ln_to_node_id = extract_nodes_with_location_info(nodes)
        code = read_code_file(os.path.join(dir_path, file))
        parsed_res_map[file] = [nodes, edges, node_key_map, node_id_to_ln, ln_to_node_id, code]


    def build_slice_graph(parsed_res_map: Dict, file_name:str, split_line: int, slice_graph: List, seen: List):
        infos = parsed_res_map[file_name]
        nodes = infos[0]
        edges = infos[1]
        node_key_map = infos[2]
        node_id_to_ln = infos[3]
        ln_to_node_id = infos[4]

        pdg= build_PDG(node_key_map, edges, node_id_to_ln)
        xfg = build_XFG(pdg, int(split_line))

        slice_criteria_node = ln_to_node_id[split_line]
        function_node_id = node_key_map[slice_criteria_node]['functionId']
        function_node = node_key_map[function_node_id]
        func_name = function_node['code']
        sub_slice_graph = {'file_name':file_name,'func_ln': node_id_to_ln[function_node['key']], 'xfg':xfg}
        if len(xfg.edges) == 0:
            return 
        slice_graph.append(sub_slice_graph)

        
        if has_arg(node_key_map,get_function_def(function_node['key'])):
            funcall_node = get_function_call(nodes, func_name) 

            if funcall_node is not None:
                build_slice_graph(parsed_res_map,file_name, node_id_to_ln[funcall_node['key']], slice_graph, seen)
            else:
                seen.append(file_name)
                for file in parsed_res_map.keys():
                    if file not in seen:
                        n = parsed_res_map[file][0]
                        funcall_node = get_function_call(n, func_name)
                        if funcall_node is not None:
                            mp = parsed_res_map[file][3]
                            build_slice_graph(parsed_res_map, file,  mp[funcall_node['key']], slice_graph, seen)
                            break
                seen.pop()
                

                
    slice_graph = []
    build_slice_graph(parsed_res_map, target_file, int(split_line), slice_graph, [])

    def connect_graph(graph: Dict, start_index: int, prev_call_index: int,total_nodes: List, total_edges:List) -> Tuple[int, int]:
        cdg = []
        ddg = []
        lines = set()
        order = []
        file_name = graph['file_name']
        func_ln = graph['func_ln']
        
        lines.add(func_ln)


        code_map = parsed_res_map[file_name][5]
        split_ln = graph['xfg'].graph["key_line"] 
        for (u, v, t) in graph['xfg'].edges.data('edge'):
            lines.add(u)
            lines.add(v)
            if t == 'c':
                cdg.append([u,v])
            else:
                ddg.append([u,v])
        
        lines = sorted(list(lines))
        window = 3
        if len(lines) >= window:
            for i in range(0, len(lines)-window+1):
                for j in range(i+1,i+window):
                    order.append([lines[i], lines[j]])
                #print(lines[i], '------'+code_map[lines[i]])
            
        ln_map = {}
        for ln in lines:
            ln_map[ln] = start_index
            start_index+=1
            code = code_map[ln]
            total_nodes.append(code)
        
        if prev_call_index is not None:
            total_edges.append([prev_call_index, 4, ln_map[func_ln]])

        for edge in order:
            total_edges.append([ln_map[edge[0]], 1, ln_map[edge[1]]])
        for edge in ddg:

            total_edges.append([ln_map[edge[0]], 2, ln_map[edge[1]]])
        for edge in cdg:
            total_edges.append([ln_map[edge[0]], 3, ln_map[edge[1]]])
       

        return start_index, ln_map[split_ln] 


    total_edges = []
    total_nodes = []
    start_index = 0
    prev_call_index = None  
    for sub_slice_graph in slice_graph[::-1]:
      start_index, prev_call_index = connect_graph(sub_slice_graph, start_index, prev_call_index , total_nodes, total_edges)
                  
    graph ={
        'node_code': total_nodes,
        'edges':total_edges,
        'label':int(label),
    }
    with open(os.path.join(target_path, id +'.json'),'w') as f:
        json.dump(graph, f)
  

    #slice_line_ids = [ln_to_node_id[ln] for ln in all_slice_lines]
    #filter_node_ids = filter_nodes(ast_edges, slice_line_ids)
    #wv = Word2Vec.load("wordvec.model")
    #grapth = inputGeneration2(slice_line_ids, filter_node_ids, nodes_path, edges_path, label, wv,edgeType_full)
   
    
        
def get_slice_statement_graph(line):
    tmp = line.split(' ')
    file_path, split_line, label = tmp[1], tmp[2].strip(), tmp[3].strip()
    file_name = "___".join(file_path.split('/'))
    os.system('rm -r  tmp' )
    os.system('mkdir -p tmp')
    os.system('rm -r  parsed' )
    source = os.path.join(base,file_path)
    os.system('cp ' + source + ' tmp/'+file_name)
    os.system('./joern/joern-parse tmp/')
    nodes_path = os.path.join('parsed', 'tmp',file_name,'nodes.csv')
    edges_path = os.path.join('parsed', 'tmp', file_name, 'edges.csv')
    nodes = read_csv(nodes_path)
    edges = read_csv(edges_path)
    code = read_code_file('tmp/'+file_name)
    node_indices, node_ids, line_numbers, node_id_to_ln, ln_to_node_id = extract_nodes_with_location_info(nodes)
    adjacency_list, ast_edges = create_adjacency_list(line_numbers, node_id_to_ln, edges, False)

    combined_graph = combine_control_and_data_adjacents(adjacency_list)          
    forward_sliced_lines = create_forward_slice(combined_graph, int(split_line))
    backward_sliced_lines = create_backward_slice(combined_graph, int(split_line))
            

    all_slice_lines = forward_sliced_lines
    all_slice_lines.extend(backward_sliced_lines)
    all_slice_lines = sorted(list(set(all_slice_lines)))

    ast = []
    cdg = []
    ddg = []
    order = []
    for i in range(1,len(all_slice_lines)):
        order.append([all_slice_lines[i-1], all_slice_lines[i]])
    for src_ln, target_ln_set in adjacency_list.items():
        if src_ln not in all_slice_lines:
            continue
        for ln in target_ln_set[0]:
            if ln in all_slice_lines:
                cdg.append([src_ln, ln])
        for ln in target_ln_set[1]:
            if ln in all_slice_lines:
                ddg.append([src_ln, ln])

    #slice_line_ids = [ln_to_node_id[ln] for ln in all_slice_lines]
    #filter_node_ids = filter_nodes(ast_edges, slice_line_ids)

    
    data ={
    'file_path': file_path,
    'cdg':cdg,
    'ddg':ddg,
    'order':order,
    'label':label,
    }
    datas.append(data)


def get_statement_graph(dir_name, target_path):

    def statement_graph(source_path, target_path):
        i = 0
        f = open(target_path, 'w') 
        
        for file in os.listdir(source_path):
            file_path = os.path.join(source_path, file)

            os.system('rm -r  tmp' )
            os.system('mkdir -p tmp')
            os.system('rm -r  parsed' )

            os.system('cp ' + file_path + ' tmp/'+file)
            os.system('./joern/joern-parse tmp/')
            nodes_path = os.path.join('parsed', 'tmp',file,'nodes.csv')
            edges_path = os.path.join('parsed', 'tmp', file, 'edges.csv')
            nodes = read_csv(nodes_path)
            edges = read_csv(edges_path)
            
            
            try:
                graph = get_statement_gtaph(nodes, edges, file_path)
                graph['file'] = file
                f.write(json.dumps(graph)+'\n')
            except:
                pass
            i+=1;
            print("------------", i)
        f.close()
            
        
        

    def slice_graph(source_path, target_path):
        i = 0
        mutil_graph_datas = []
        for file in os.listdir(source_path):
            file_path = os.path.join(source_path, file)

            os.system('rm -r  tmp' )
            os.system('mkdir -p tmp')
            os.system('rm -r  parsed' )
            
            os.system('cp ' + file_path + ' tmp/'+file)
            os.system('./joern/joern-parse tmp/')
            nodes_path = os.path.join('parsed', 'tmp',file,'nodes.csv')
            edges_path = os.path.join('parsed', 'tmp', file, 'edges.csv')
            nodes = read_csv(nodes_path)
            edges = read_csv(edges_path)
            
            
            try:
                data = get_slice(nodes, edges, file_path)
                mutil_graph_datas.append(data)
            except:
                pass
            i+=1;
            print("------------", i)
            
        with open(target_path, 'w') as f:
            json.dump(mutil_graph_datas, f)
    
    statement_graph(dir_name, target_path)

            
        
def get_slice_code_tokens(line: str):

    tmp = line.split(' ')
    file_path, split_line, label = tmp[1], tmp[2].strip(), tmp[3].strip()
    file_name = "___".join(file_path.split('/'))
    os.system('rm -r  tmp' )
    os.system('mkdir -p tmp')
    os.system('rm -r  parsed' )
    source = os.path.join(base,file_path)
    os.system('cp ' + source + ' tmp/'+file_name)
    os.system('./joern/joern-parse tmp/')
    nodes_path = os.path.join('parsed', 'tmp',file_name,'nodes.csv')
    edges_path = os.path.join('parsed', 'tmp', file_name, 'edges.csv')
    nodes = read_csv(nodes_path)
    edges = read_csv(edges_path)
    code = read_code_file('tmp/'+file_name)
    node_key_map, node_ids, line_numbers, node_id_to_ln, ln_to_node_id = extract_nodes_with_location_info(nodes)
    adjacency_list, ast_edges = create_adjacency_list(line_numbers, node_id_to_ln, edges, False)
    combined_graph = combine_control_and_data_adjacents(adjacency_list)          
    forward_sliced_lines = create_forward_slice(combined_graph, int(split_line))
    backward_sliced_lines = create_backward_slice(combined_graph, int(split_line))

    all_slice_lines = forward_sliced_lines
    all_slice_lines.extend(backward_sliced_lines)
    all_slice_lines = sorted(list(set(all_slice_lines)))

    txt_tokens = []
    for ln_num in all_slice_lines:
        tokens = code[ln_num].split(' ')
        for token in tokens:
            if token != '':
                parse_token(token, txt_tokens)
    data = {
        'file_path':tmp[0],
        'code': " ".join(txt_tokens),
        'label': int(label),
    }
    
    return data
        

if __name__ == '__main__':
    import os
    from xml.dom.minidom import parse
    import csv
    #data = get_slice_code_tokens("82924 100954/CWE404_Improper_Resource_Shutdown__fopen_w32_close_84_goodB2G.cpp fclose 35")
    #get_statement_graph('/home/zhangxs/data/Data/Asterisk')
    #175542 150410/gimpdisplayshell-appearance.c fgetc 142
    """  get_statement_graph('dataset/t', 'dataset/t_graph.json')
    get_statement_graph('dataset/test', 'dataset/test_graph.json')
    get_statement_graph('dataset/train', 'dataset/train_graph.json')
    get_statement_graph('dataset/valid', 'dataset/valid_graph.json') """
    #datafile = csv.DictWriter(open('test_code_tokens.csv', 'w', encoding='utf-8'), fieldnames=header)

    
    #get_slice_graph('1777 117306/CWE789_Uncontrolled_Mem_Alloc__malloc_wchar_t_fgets_63b.c wcslen 38 0', '../data/graph/test')
    i = 0
    for file in os.listdir("cve"):
        if not file.endswith('txt'):
            continue
        dir_path = os.path.join('cve', file.split('.')[0])
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        fail_log = open("fail.log",'w')
        with open(os.path.join('cve', file), 'r') as f:
            
            for line in f.readlines():
                try:
                    i+=1;
                    
                    get_slice_graph(line, dir_path)
                    #data = get_slice_code_tokens(line)
                    #datafile.writerow(data)
                    #get_slice_token_graph(line)
                    #get_slice_statement_graph(line)
                    print("----------------count: ",i);
                except:
                    fail_log.write(line)
                    
        fail_log.close()  
    """ with open("train_slice_statement_graph.json", 'w') as fp:
            json.dump(datas, fp)
            fp.close() """

                
    """ 
    path = "/home/zhangxs/data/source-code"
    dirs = os.listdir(path)

    file_line_map ={}
    for f1 in dirs:
        if not os.path.isdir(os.path.join(path, f1)):
            continue
        dirs2 = os.listdir(os.path.join(path, f1))
        for f2 in dirs2:
            if f2.endswith('xml'):
                dom = parse(os.path.join(path, f1, f2))
                rootnode = dom.documentElement
                testcases = rootnode.getElementsByTagName('testcase')
                for testcase in testcases:
                    
                    files = testcase.getElementsByTagName('file')
                    for file in files:
                        mixed = file.getElementsByTagName('mixed')
                        if len(mixed) == 0 :
                            continue
                        mixed = mixed[0]                      
                        file_line_map[os.path.join(path, f1, 'testcases',file.getAttribute('path'))] = mixed.getAttribute('line') 
    directory = 'tmp'
    output = 'out1'
    #file_line_map['/home/zhangxs/data/source-code/119-12300-c/testcases/000/062/731/CWE121_Stack_Based_Buffer_Overflow__CWE129_listen_socket_34.c'] = 76
    #file_line_map['/home/zhangxs/data/source-code/119-cpp/testcases/000/067/735/CWE122_Heap_Based_Buffer_Overflow__cpp_CWE129_listen_socket_34.cpp'] = 143
     
    for file_path, line in file_line_map.items():
        
        tmp = file_path.split('/')
        f = tmp[-1]
        os.system('rm -r  tmp' )
        os.system('mkdir -p tmp')
        os.system('cp ' +  file_path + ' tmp/' + f)
        os.system('rm -r  parsed' )
        os.system('./joern/joern-parse tmp/')
        nodes_path = os.path.join('parsed', directory, f,'nodes.csv')
        edges_path = os.path.join('parsed', directory, f, 'edges.csv')
        nodes = read_csv(nodes_path)
        edges = read_csv(edges_path)
        code = read_code_file(file_path)
        node_indices, node_ids, line_numbers, node_id_to_ln = extract_nodes_with_location_info(nodes)
        adjacency_list = create_adjacency_list(line_numbers, node_id_to_ln, edges, False)
        combined_graph = combine_control_and_data_adjacents(adjacency_list)
        if not os.path.exists(output):
            os.mkdir(output)
        forward_sliced_lines = create_forward_slice(combined_graph, int(line))
        backward_sliced_lines = create_backward_slice(combined_graph, int(line))
        output_path = os.path.join(output, f + '.slice')
        slice_lines = backward_sliced_lines + forward_sliced_lines
        

        
        fp = open(output_path, 'w')
        for ln in backward_sliced_lines:
            fp.write(code[ln] + ' '+ str(ln)+'\n')
        for ln in forward_sliced_lines:
            fp.write(code[ln] + ' '+ str(ln)+'\n')        
        fp.close()"""


    """ parser = argparse.ArgumentParser()
    parser.add_argument('--code', help='Name of code file', default='test1.c')
    parser.add_argument('--line', help='Line Number for slice start point', type=int, default=22)
    parser.add_argument('--data_flow_only', action='store_true', help='Slice only on data flow graph.')
    parser.add_argument('--output', help='Output where slice results will be stored.', default='slice-output')
    parser.add_argument('--verbose', help='Show the slice results and the graph.', action='store_true')
    args = parser.parse_args()
    directory = 'tmp'
    file_name = args.code
    slice_ln = int(args.line)
    code_file_path = os.path.join(directory, file_name)
    nodes_path = os.path.join('parsed', directory, file_name, 'nodes.csv')
    edges_path = os.path.join('parsed', directory, file_name, 'edges.csv')
    nodes = read_csv(nodes_path)
    edges = read_csv(edges_path)
    #print(nodes)
    #print(edges)
    code = read_code_file(code_file_path)
    node_indices, node_ids, line_numbers, node_id_to_ln = extract_nodes_with_location_info(nodes)
    adjacency_list = create_adjacency_list(line_numbers, node_id_to_ln, edges, args.data_flow_only)
    #print(adjacency_list)
    #create_visual_graph(code, adjacency_list, os.path.join(args.output, file_name), verbose=args.verbose)
    combined_graph = combine_control_and_data_adjacents(adjacency_list)

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    forward_sliced_lines = create_forward_slice(combined_graph, slice_ln)
    forward_output_path = os.path.join(args.output, file_name + '.forward')
    fp = open(forward_output_path, 'w')
    for ln in forward_sliced_lines:
        fp.write(code[ln] + '\n')
    fp.close()

    backward_sliced_lines = create_backward_slice(combined_graph, slice_ln)
    backward_output_path = os.path.join(args.output, file_name + '.backward')
    fp = open(backward_output_path, 'w')
    for ln in backward_sliced_lines:
        fp.write(code[ln] + '\n')
    fp.close() """
""" 
    if args.verbose:
        print('============== Actual Code ====================')
        for ln in sorted(set(line_numbers)):
            print(ln, '\t->', code[ln])
        print('===============================================')
        print('\n\nStarting slice for line', slice_ln)
        print('-----------------------------------------------')
        print(code[slice_ln])
        print('-----------------------------------------------')
        print('============== Forward Slice ==================')
        for ln in forward_sliced_lines:
            print(ln, '\t->', code[ln])
        print('===============================================')

        print('============== Backward Slice =================')
        for ln in backward_sliced_lines:
            print(ln, '\t->', code[ln])
        print('===============================================')

    pass """