import json
import torch

# ------------ get system and user prompts -----------------
def get_prompts():
    # we can set the prompts manually, or better, load them from a given file,
    # so we can experiment with different prompt combinations without changing the code

    # EXTREU-RE'LS D'UN JSON

    prompts = {}
    prompts["sysprompt"] = "\n".join(["You are an journalist assistant in a news agency and your mission is to read news items and extract the list of entities mentioned in it.",
                   "Mentioned entities may belong to one of the following types:",
                   "",
                   "  - Person (PER): The name of a person",
                   "  - Location (LOC): The name of a country, city, region, etc.",
                   "  - Organization (ORG): The name of a company, government agency, foundation, bank, ministery, etc."
                  ])
                  
    prompts["usrprompt"] = "\n".join(["Given the following text, extract entities, classified by their type (PER, LOC, ORG)",
               "Provide the output as a json dictionary, with a key for each appearing entity type, and a list of names as value (e.g. {\"PER\":[\"John Smith\"], \"LOC\":[\"USA\", \"Berlin\"]} )",
	       "",
               "It is important that you take into account the following constraints:",
               "  - If an entity is mentioned twice with the same words, extract it only once.",
               "  - If an entity is mentioned twice with the different words (e.g. John Smith and Mr. Smith), extract it in both cases.",
               "  - Only the three types in the above list are valid. DO NOT introduce new ones.",
               "  - DO NOT INCLUDE in the json entity types for which no entity is mentioned.",
               "  - Produce just the json, do not add further explanations."
              ])

    return prompts
    
       
# ------------ load dataset (either "train", or "devel") -----------------
def load_dataset(data) :
    with open(f"dataset/{data}.json") as f : 
        examples = json.load(f)
    return examples

# ------------ prepare prompt messages for an example ----------------
def prepare_messages(prompts, question) :
    
    # First, system message
    messages = [{"role": "system", "content": prompts["sysprompt"]}]
 
    # then, add the text we want the anwer for, so the model will complete the response
    messages.append({"role": "user", 
                     "content": prompts["usrprompt"]
                                + "\nTEXT: "
                                + question['text']})
    return messages




# ------------ add review to prompt, tokenize, and encode ----------------
def encode(tokenizer, messages) :
    # convert the sequence of messages to an actual prompt in the format
    # expected by the model.
    # Tokenize and encode the prompt.
    input_ids = tokenizer.apply_chat_template(messages,
                                              tokenize=True,
                                              add_generation_prompt=True,
                                              return_tensors="pt").to("cuda")
    return input_ids
    
    
    
# ------------ generate completion for given tokens ----------------
def generate(model, tokenizer, input_ids):
    # generate likely continuation (assistant answer)
    with torch.no_grad():
        gen_tokens = model.generate(input_ids,
                                    max_new_tokens=256,
                                    pad_token_id=tokenizer.eos_token_id, # if there are two or more sequences in the batch, they may have different lengths, so we need to pad them to the same length (we add EOS token to fill the blanks). We use the eos_token_id for padding, so it will not affect the generation
                                    do_sample=False, # we want the most likely continuation, so we set do_sample to False, which means that the model will not sample from the distribution of possible continuations, but will always choose the one with the highest probability. If we set it to True, the model would sample from the distribution, which could lead to more diverse but also less accurate responses.
                                    temperature=0.7,
                                    top_p=0.9
                                   )
    promptlen = len(input_ids[0]) # input_ids is a batch of sequences [ [13, 543, 54], [342, 432, 264]... ], but we have only one sequence in the batch, so we take the length of the first one. We need this to know where the prompt ends and the generated text starts in gen_tokens.
    # decode obtained tokens back into text
    gen_text = tokenizer.decode(gen_tokens[0][promptlen:], skip_special_tokens=True)
    return gen_text

# ------------  # find '}' matching '{' in given position p ----------------
def find_matching_bracket(gen_text, p) :
    q = p+1
    c = 1
    while c!=0 and q<len(gen_text) :  # find '}' matching first '{'
       if gen_text[q]=='{' : c += 1
       elif gen_text[q]=='}' : c -= 1
       q += 1
    return q-1

# ------------ clean up response, extracting just the expected json part ----------------
def extract_json(gen_text):    

    # find first '{' and its matching closed bracket '}'
    p = gen_text.find("{") 
    q = find_matching_bracket(gen_text, p)
           
    # get text between first '{' and its matching '}'
    predic = gen_text[p:q+1].replace('“','"').replace('”','"')
    try :
        # load string as json object
        prediction = json.loads(predic)
    except json.JSONDecodeError as err:
        # json was missing or wrongly formatted
        print(repr(err))
        print(gen_text)
        prediction = {}
    return prediction

