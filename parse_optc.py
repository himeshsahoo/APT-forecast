import os
import ujson as json
import pandas as pd

# --- THREAT INTELLIGENCE DICTIONARY ---
# Extracted from OpTCRedTeamGroundTruth.pdf
COMPROMISED_HOSTS = [
    "sysclient0201", "sysclient0402", "sysclient0660", "dc1.systemia.com", 
    "sysclient0501", "sysclient0811", "sysclient0005", "sysclient0974", 
    "sysclient0051", "sysclient0351"
]

MALICIOUS_KEYWORDS = [
    "runme.bat", "mimikatz", "psinject", "invoke_wmi", "zipfidr.dll", 
    "payroll.docx", "transfer1000.exe", "nc.exe", "export.zip", 
    "allgone.zip", "movingonup.exe", "update.exe", "ckfgw.exe", 
    "myhbyxtpviwx.vbx", "deathstar"
]

def check_if_malicious(hostname, actor_process, target_process):
    """Flags the event as 1 if it matches known Red Team IoCs."""
    host = str(hostname).lower()
    actor = str(actor_process).lower()
    target = str(target_process).lower()
    
    # 1. Is the event happening on a known breached machine?
    if not any(ch in host for ch in COMPROMISED_HOSTS):
        return 0
        
    # 2. Does the event involve a known malicious file or command?
    if any(mk in actor for mk in MALICIOUS_KEYWORDS) or any(mk in target for mk in MALICIOUS_KEYWORDS):
        return 1
        
    return 0

def parse_ecar_log(input_file_path, output_csv_path, max_lines=100000):
    parsed_events = []
    print(f"Starting to parse: {input_file_path}")
    
    with open(input_file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                print(f"\nReached limit of {max_lines} lines for testing.")
                break
                
            try:
                event = json.loads(line.strip())
                props = event.get("properties", {})
                
                obj = event.get('object', 'UNKNOWN')
                act = event.get('action', 'UNKNOWN')
                
                hostname = event.get("hostname", "None")
                actor = props.get("parent_image_path", "None")
                target = props.get("image_path", "None")
                
                extracted_data = {
                    "timestamp": event.get("timestamp"),
                    "hostname": hostname,
                    "action": f"{obj}_{act}", 
                    "actor_process": actor,
                    "target_process": target,
                    "dest_ip": props.get("dest_ip", "None"),
                    "dest_port": props.get("dest_port", "None"),
                    "is_malicious": check_if_malicious(hostname, actor, target)
                }
                
                parsed_events.append(extracted_data)
                
            except Exception as e:
                continue
                
            if i % 25000 == 0 and i > 0:
                print(f"Processed {i} lines...")

    df = pd.DataFrame(parsed_events)
    df.to_csv(output_csv_path, index=False)
    
    print(f"\nSuccessfully saved parsed data to: {output_csv_path}")
    print(f"Total Malicious Events Found: {df['is_malicious'].sum()}")

if __name__ == "__main__":
    # Ensure this path still points to your extracted .json file
    INPUT_JSON = r"C:\Users\Himes\Desktop\apt-forecast-project\AIA-1-25.ecar-last.json\AIA-1-25.ecar.json"
    OUTPUT_CSV = "optc_parsed_sequences.csv"
    
    # Let's increase the max_lines to 500,000 to ensure we catch some attacks!
    parse_ecar_log(INPUT_JSON, OUTPUT_CSV, max_lines=500000)