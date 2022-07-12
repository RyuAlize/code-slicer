import json
import os

def run(path, dir):
    sentenses=[]
    with open(path, 'r') as f:
        for line in f.readlines():
            data = json.loads(line.strip('\n'))
            func = data['func']
            lines  = [line for line in func.split('\n') if line != "" ]
            
            codelines = []
            j = 0
            while j < len(lines):
                code = lines[j]    
                if '//' in code:
                    code = code[:code.find('//')]             
                while lines[j].endswith(','):
                    code += ' '+lines[j+1].strip(" ")
                    j += 1        
                if code != "":   
                    codelines.append(code+'\n')
                j += 1
            
            with open(os.path.join(dir, data['project']+data['commit_id']+'.c'), 'w') as fp:
                fp.writelines(codelines)


run('dataset/test.jsonl', 'dataset/test')
run('dataset/valid.jsonl', 'dataset/valid')
run('dataset/train.jsonl', 'dataset/train')   