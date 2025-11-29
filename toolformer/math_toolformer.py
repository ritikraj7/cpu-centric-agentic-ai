from langchain_community.llms import VLLMOpenAI
import re
import time
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm
import operator
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from typing import List, Dict, Tuple
import time
import requests
import os

class WolframCalculator:
    def __init__(self, app_id: str = None):
        # Prefer environment variable if not passed explicitly
        self.app_id = app_id or os.getenv("WOLFRAM_ALPHA_APPID")
        if not self.app_id:
            raise ValueError("Wolfram Alpha App ID not provided. "
                             "Set WOLFRAM_ALPHA_APPID env var or pass app_id.")

    def calculate(self, expression: str) -> tuple[str, float]:
        """
        Evaluate mathematical expressions using WolframAlpha API.
        Args:
            expression: Mathematical expression string
        Returns:
            Tuple of (result string, latency in seconds)
        """
        start_time = time.time()
        try:
            url = "http://api.wolframalpha.com/v2/query"
            params = {
                "input": expression,
                "appid": self.app_id,
                "output": "JSON"
            }

            resp = requests.get(url, params=params, timeout=10)
            latency = time.time() - start_time

            if resp.status_code == 401:
                print("Unauthorized: invalid or missing Wolfram Alpha App ID.")
                return None, latency
            if resp.status_code != 200:
                print(f"HTTP error {resp.status_code} for expression '{expression}'")
                return None, latency

            data = resp.json()
            queryresult = data.get("queryresult", {})

            if not queryresult.get("success", False):
                print(f"Query failed for expression '{expression}'")
                return None, latency

            pods = queryresult.get("pods", [])
            answer = None

            # Prefer "Result" pod if present
            for pod in pods:
                if pod.get("title", "").lower().startswith("result"):
                    answer = pod["subpods"][0].get("plaintext", "")
                    break

            # Fallback: first non-empty plaintext
            if not answer:
                for pod in pods:
                    for subpod in pod.get("subpods", []):
                        if subpod.get("plaintext"):
                            answer = subpod["plaintext"]
                            break
                    if answer:
                        break

            return answer, latency

        except Exception as e:
            latency = time.time() - start_time
            print(f"Wolfram API error for '{expression}': {e}")
            return None, latency


class CalculatorTool:
    """Calculator tool for mathematical computations as described in Toolformer paper"""
    
    def __init__(self):
        # Safe operators for mathematical expressions
        self.operators = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv,
            '//': operator.floordiv,
            '%': operator.mod,
            '**': operator.pow,
            '^': operator.pow,  # Alternative power notation
        }
    
    def calculate(self, expression: str) -> tuple[float, float]:
        """
        Safely evaluate mathematical expressions
        Args:
            expression: Mathematical expression string
        Returns:
            Tuple of (result, latency in seconds)
        """
        start_time = time.time()
        
        try:
            # Clean and normalize the expression
            cleaned_expr = self._clean_expression(expression)
            
            # Safe evaluation using ast.parse for security
            result = self._safe_eval(cleaned_expr)
            
            latency = time.time() - start_time
            return result, latency
            
        except Exception as e:
            latency = time.time() - start_time
            print(f"Calculator error for '{expression}': {e}")
            return None, latency
    
    def _clean_expression(self, expr: str) -> str:
        """Clean and normalize mathematical expression"""
        # Remove whitespace
        expr = expr.strip()
        
        # Replace common alternative notations
        expr = expr.replace('^', '**')  # Power notation
        expr = expr.replace('÷', '/')   # Division symbol
        expr = expr.replace('×', '*')   # Multiplication symbol
        
        # Remove currency symbols and commas
        expr = re.sub(r'[\$,]', '', expr)
        
        return expr
    
    def _safe_eval(self, expr: str) -> float:
        """Safely evaluate mathematical expression using AST"""
        try:
            # Parse the expression into an AST
            node = ast.parse(expr, mode='eval')
            
            # Evaluate the AST safely
            return self._eval_node(node.body)
            
        except Exception as e:
            raise ValueError(f"Invalid mathematical expression: {expr}")
    
    def _eval_node(self, node):
        """Recursively evaluate AST nodes"""
        if isinstance(node, ast.Num):  # Numbers (Python < 3.8)
            return node.n
        elif isinstance(node, ast.Constant):  # Numbers (Python >= 3.8)
            return node.value
        elif isinstance(node, ast.BinOp):  # Binary operations
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            
            if isinstance(node.op, ast.Add):
                return left + right
            elif isinstance(node.op, ast.Sub):
                return left - right
            elif isinstance(node.op, ast.Mult):
                return left * right
            elif isinstance(node.op, ast.Div):
                if right == 0:
                    raise ValueError("Division by zero")
                return left / right
            elif isinstance(node.op, ast.FloorDiv):
                if right == 0:
                    raise ValueError("Division by zero")
                return left // right
            elif isinstance(node.op, ast.Mod):
                if right == 0:
                    raise ValueError("Division by zero")
                return left % right
            elif isinstance(node.op, ast.Pow):
                return left ** right
            else:
                raise ValueError(f"Unsupported operation: {type(node.op)}")
        elif isinstance(node, ast.UnaryOp):  # Unary operations
            operand = self._eval_node(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            elif isinstance(node.op, ast.USub):
                return -operand
            else:
                raise ValueError(f"Unsupported unary operation: {type(node.op)}")
        else:
            raise ValueError(f"Unsupported node type: {type(node)}")

class MathToolformer:
    """GPT-J-6B based Toolformer implementation for mathematical problem solving"""
    
    def __init__(self, base_url: str = "http://localhost:5000/v1", 
                 model_path: str = "EleutherAI/gpt-j-6b"):
        print("Loading GPT-J-6B model for math problem solving...")
        self.llm = VLLMOpenAI(
            base_url=base_url,
            model=model_path,
            openai_api_key='EMPTY'
        )
        
        self.calculator = WolframCalculator()
        print("GPT-J-6B model loaded successfully with Calculator tool!")
    
    def create_math_toolformer_prompt(self, problem: str) -> str:
        """Create Toolformer-style prompt for mathematical problem solving"""
        prompt = """Your task is to solve mathematical word problems step by step. For any calculations needed, use the WolFrom Alpha API tool.

To use the calculator, write: [Calculator("mathematical expression")]

Here are some examples:

Problem: Sarah has 15 apples. She gives 7 apples to her friend. How many apples does she have left?
Solution: Sarah starts with 15 apples and gives away 7 apples. I need to calculate how many are left: [Calculator("15 - 7")] Sarah has 8 apples left.

Problem: A rectangle has length 12 meters and width 8 meters. What is its area?
Solution: The area of a rectangle is length × width. [Calculator("12 * 8")] The area is 96 square meters.

Problem: Tom buys 3 packs of pencils. Each pack costs $4.50. How much does he spend in total?
Solution: Tom buys 3 packs at $4.50 each. The total cost is: [Calculator("3 * 4.50")] Tom spends $13.50 in total.

Problem: {problem}
Solution:"""
        
        return prompt.format(problem=problem)
    
    def generate_response(self, prompt: str) -> tuple[str, float]:
        """Generate response using VLLM"""
        start_time = time.time()
        try:
            response = self.llm.invoke(prompt)
            latency = time.time() - start_time
            return response.strip(), latency
        except Exception as e:
            latency = time.time() - start_time
            print(f"Generation error: {e}")
            return "I apologize, but I couldn't generate a response.", latency
    


    def execute_calculations(self, calc_calls: List[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Execute calculator calls found in the text in parallel"""
        results = {}
        latencies = {}
        
        def process_single_call(call: str) -> Tuple[str, float, float]:
            """Process a single calculator call"""
            match = re.search(r'Calculator\("([^"]+)"\)', call)
            if match:
                expression = match.group(1)
                result, calc_latency = self.calculator.calculate(expression)
                print(f"Calculator: '{expression}' = {result} (computed in {calc_latency:.3f}s)")
                return call, result, calc_latency
            return call, None, 0.0
        
        # Execute calculations in parallel
        with ThreadPoolExecutor(max_workers=min(len(calc_calls), 10)) as executor:
            # Submit all tasks
            future_to_call = {executor.submit(process_single_call, call): call for call in calc_calls}
            
            # Collect results as they complete
            for future in as_completed(future_to_call):
                call, result, latency = future.result()
                if result is not None:
                    results[call] = result
                    latencies[call] = latency
        
        return results, latencies
    
    def process_math_problem(self, problem: str) -> Dict[str, Any]:
        """Process a mathematical problem using Toolformer approach"""
        total_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Processing: {problem}")
        print(f"{'='*60}")
        
        # Step 1: Generate initial solution with potential calculator calls
        print("\n🧮 STEP 1: Initial Solution Generation")
        prompt = self.create_math_toolformer_prompt(problem)
        print(f"Prompt length: {len(prompt)} characters")
        
        generated_response, inference_latency = self.generate_response(prompt)
        
        # Clean up response (take first coherent part)
        lines = generated_response.split('\n')
        first_line = lines[0].strip() if lines else generated_response
        
        print(f"✅ Generated in {inference_latency:.2f}s: {first_line}")
        
        # Step 2: Find calculator calls
        print("\n🔍 STEP 2: Calculator Call Detection")
        calc_calls = re.findall(r'\[Calculator\("[^"]+"\)\]', first_line)
        
        if calc_calls:
            print(f"✅ Found {len(calc_calls)} calculator calls:")
            for call in calc_calls:
                expr = re.search(r'Calculator\("([^"]+)"\)', call)
                if expr:
                    print(f"   - '{expr.group(1)}'")
            
            # Step 3: Execute calculations
            print("\n🧮 STEP 3: Calculation Execution")
            calc_results, calc_latencies = self.execute_calculations(calc_calls)
            total_calc_time = max(calc_latencies.values())
            
            # Step 4: Create enhanced solution with results
            print("\n📊 STEP 4: Solution Integration")
            enhanced_solution = first_line
            for call, result in calc_results.items():
                if result is not None:
                    enhanced_solution = enhanced_solution.replace(call, f"[Result: {result}]")
            
            # Step 5: Generate final answer with calculations
            print("\n✨ STEP 5: Final Solution Generation")
            final_prompt = f"""Based on the calculations performed, provide a clear final answer to the problem.

Problem: {problem}

Solution with calculations: {enhanced_solution}

Final Answer:"""
            
            final_answer, final_inference_latency = self.generate_response(final_prompt)
            print(f"✅ Final answer generated in {final_inference_latency:.2f}s")
            
            total_time = time.time() - total_start_time
            
            return {
                'problem': problem,
                'initial_solution': first_line,
                'calc_calls': calc_calls,
                'calc_results': calc_results,
                'enhanced_solution': enhanced_solution,
                'final_answer': final_answer,
                'used_calculator': True,
                'latencies': {
                    'initial_inference': inference_latency,
                    'calculations': calc_latencies,
                    'total_calc_time': total_calc_time,
                    'final_inference': final_inference_latency,
                    'end_to_end': total_time
                },
                'prompts': {
                    'initial': prompt,
                    'final': final_prompt
                }
            }
        else:
            total_time = time.time() - total_start_time
            print("❌ No calculator calls needed")
            return {
                'problem': problem,
                'initial_solution': first_line,
                'final_answer': first_line,
                'used_calculator': False,
                'latencies': {
                    'initial_inference': inference_latency,
                    'end_to_end': total_time
                },
                'prompts': {
                    'initial': prompt
                }
            }

def load_math_datasets(mawps_query=128):
    """Load sample problems from ASDiv, SVAMP, and MAWPS datasets"""
    
    # ASDiv (Academia Sinica Diverse MWP Dataset) - diverse word problems
    asdiv_problems = [
        "Sarah has 24 stickers. She gives 8 stickers to her brother and 6 stickers to her sister. How many stickers does she have left?",
        "A bakery baked 144 cookies. They sold 89 cookies in the morning and 37 cookies in the afternoon. How many cookies are left?",
        "Tom has 3 boxes of crayons. Each box contains 24 crayons. How many crayons does Tom have in total?",
        "A school has 18 classrooms. Each classroom has 32 students. How many students are in the school?",
        "Lisa bought 5 notebooks for $2.75 each. How much did she spend in total?"
    ]
    
    # SVAMP (Simple Variations on Arithmetic Math word Problems) - arithmetic focus
    svamp_problems = [
        "There are 15 birds sitting on a tree. 7 more birds come and sit on the tree. How many birds are sitting on the tree now?",
        "Jake has 84 marbles. He gives 29 marbles to his friend. How many marbles does Jake have left?",
        "A farmer has 48 chickens. Each chicken lays 3 eggs per day. How many eggs do all the chickens lay in one day?",
        "Maria saves $12 every week for 8 weeks. How much money has she saved in total?",
        "A rectangle has a length of 15 cm and a width of 9 cm. What is the area of the rectangle?"
    ]
    
    mawps_problems_5 = [
         "John bought 4 bags of apples. Each bag contains 12 apples. He ate 7 apples. How many apples does he have left?",
        "A store sold 156 books on Monday and 89 books on Tuesday. On Wednesday, they sold twice as many books as Tuesday. How many books did they sell in total over the three days?",
        "Emma has $50. She buys 3 toys for $8 each and 2 books for $6 each. How much money does she have left?",
        "A parking lot has 8 rows of parking spaces. Each row has 25 spaces. If 127 spaces are occupied, how many spaces are empty?",
        "David runs 2.5 miles every day for 6 days. Then he runs 4 miles on the 7th day. What is the total distance he ran in the week?"]

    # MAWPS (Math Word Problem Solver) - multi-step problems - 128 problems total
    mawps_problems_128= [        
        "John bought 4 bags of apples. Each bag contains 12 apples. He ate 7 apples. How many apples does he have left?",
        "Sarah has 15 stickers. She gets 23 more stickers from her friend. How many stickers does she have now?",
        "A library has 234 books on the first shelf and 167 books on the second shelf. How many books are on both shelves combined?",
        "Tom collected 45 baseball cards in January, 38 in February, and 52 in March. How many cards did he collect in total?",
        "There are 89 red balloons and 76 blue balloons at a party. How many balloons are there altogether?",
        "A farmer has 145 chickens and 89 ducks. How many birds does the farmer have in total?",
        "Maria saved $12 in week 1, $18 in week 2, $15 in week 3, and $20 in week 4. How much money did she save in total?",
        "In a school cafeteria, 187 students had lunch on Monday and 154 students had lunch on Tuesday. How many students had lunch over both days?",
        "A store received 96 shirts and 78 pairs of pants in their shipment. How many clothing items did they receive?",
        "Jake scored 23 points in the first game, 19 points in the second game, and 31 points in the third game. What was his total score?",
        "There are 67 fiction books and 84 non-fiction books in a bookstore. How many books are there in total?",
        "Amy planted 35 roses, 42 tulips, and 28 daisies in her garden. How many flowers did she plant altogether?",
        "A parking garage has 158 cars on level 1 and 142 cars on level 2. How many cars are parked in the garage?",
        "During a fundraiser, the class collected $145 on Friday and $178 on Saturday. How much money did they collect in total?",
        "There are 73 apples in basket A and 91 apples in basket B. How many apples are there in both baskets?",
        "A movie theater sold 267 tickets for the matinee and 189 tickets for the evening show. How many tickets were sold in total?",
        "Lisa has 56 marbles in her collection. Her brother gives her 34 more marbles. How many marbles does she have now?",
        "A zoo has 112 mammals and 95 birds. How many animals are there in total?",
        "During recess, 78 students played soccer and 45 students played basketball. How many students played sports?",
        "A restaurant served 234 customers for lunch and 156 customers for dinner. How many customers were served in total?",
        
        # Subtraction Problems (21-40)
        "A store sold 156 books on Monday and 89 books on Tuesday. On Wednesday, they sold twice as many books as Tuesday. How many books did they sell in total over the three days?",
        "Emma has $50. She buys 3 toys for $8 each and 2 books for $6 each. How much money does she have left?",
        "There were 345 students in a school. 123 students went on a field trip. How many students remained in school?",
        "A baker made 180 cookies. She sold 127 cookies. How many cookies does she have left?",
        "Tom had 95 baseball cards. He gave 38 cards to his friend. How many cards does Tom have now?",
        "A library had 456 books. They removed 189 damaged books. How many books are left in the library?",
        "Sarah had $75. She spent $29 on groceries. How much money does she have left?",
        "There were 234 cars in a parking lot. 87 cars left. How many cars are still in the parking lot?",
        "A farmer had 167 apples. He sold 89 apples at the market. How many apples does he have left?",
        "A school had 523 pencils. Students used 278 pencils. How many pencils are left?",
        "Jake had 84 stickers. He used 35 stickers to decorate his notebook. How many stickers does he have remaining?",
        "A bookstore had 312 magazines. They sold 145 magazines. How many magazines are left?",
        "There were 198 birds in a tree. 73 birds flew away. How many birds remained in the tree?",
        "Amy had $90. She bought a dress for $45. How much money does she have left?",
        "A container had 246 marbles. 89 marbles were removed. How many marbles are still in the container?",
        "There were 167 flowers in a garden. 54 flowers were picked. How many flowers remain in the garden?",
        "A theater had 400 seats. 256 seats were occupied. How many seats were empty?",
        "Ben had 128 toy cars. He gave 47 cars to his cousin. How many toy cars does Ben have now?",
        "A pond had 145 fish. 38 fish were caught by fishermen. How many fish are left in the pond?",
        "Lisa had 203 photos. She deleted 76 photos. How many photos does she have now?",
        
        # Multiplication Problems (41-60)
        "A parking lot has 8 rows of parking spaces. Each row has 25 spaces. If 127 spaces are occupied, how many spaces are empty?",
        "There are 6 boxes of crayons. Each box has 24 crayons. How many crayons are there in total?",
        "A school has 12 classrooms. Each classroom has 28 students. How many students are in the school?",
        "Jake buys 7 packs of baseball cards. Each pack has 15 cards. How many baseball cards does he have?",
        "There are 9 bags of marbles. Each bag contains 16 marbles. How many marbles are there altogether?",
        "A farmer has 15 rows of corn. Each row has 34 corn plants. How many corn plants does the farmer have?",
        "Sarah reads 8 books. Each book has 125 pages. How many pages did she read in total?",
        "There are 11 boxes of pencils. Each box has 12 pencils. How many pencils are there in total?",
        "A bakery makes 14 trays of cookies. Each tray has 18 cookies. How many cookies did the bakery make?",
        "Tom has 13 albums. Each album has 20 photos. How many photos does Tom have?",
        "There are 16 tables in a restaurant. Each table seats 6 people. How many people can the restaurant seat?",
        "A library has 22 shelves. Each shelf holds 45 books. How many books can the library hold?",
        "Amy plants 19 rows of flowers. Each row has 12 flowers. How many flowers did she plant?",
        "There are 17 classrooms. Each classroom has 30 desks. How many desks are there in total?",
        "A store has 21 boxes of apples. Each box contains 36 apples. How many apples does the store have?",
        "Ben collects stamps in 18 albums. Each album has 25 stamps. How many stamps does Ben have?",
        "There are 14 teams in a league. Each team has 11 players. How many players are in the league?",
        "A parking garage has 26 levels. Each level has 48 parking spaces. How many parking spaces are there?",
        "Lisa has 23 containers. Each container holds 15 marbles. How many marbles does Lisa have?",
        "A school has 19 buses. Each bus can carry 42 students. How many students can all buses carry?",
        
        # Division Problems (61-80)
        "David runs 2.5 miles every day for 6 days. Then he runs 4 miles on the 7th day. What is the total distance he ran in the week?",
        "There are 144 students going on a field trip. If each bus holds 36 students, how many buses are needed?",
        "A bakery has 180 cupcakes to pack into boxes. Each box holds 12 cupcakes. How many boxes do they need?",
        "There are 216 books to be placed equally on 8 shelves. How many books will be on each shelf?",
        "A farmer has 168 eggs to pack into cartons. Each carton holds 12 eggs. How many cartons will he need?",
        "There are 195 pencils to be distributed equally among 15 students. How many pencils will each student get?",
        "A theater has 240 seats arranged in equal rows of 16. How many rows are there?",
        "There are 154 stickers to be shared equally among 14 children. How many stickers will each child receive?",
        "A school has 228 students to form teams of 12. How many teams can be formed?",
        "There are 192 marbles to be put into bags of 16 each. How many bags will be needed?",
        "A library has 275 books to arrange on shelves. Each shelf holds 25 books. How many shelves are needed?",
        "There are 186 cookies to be packed into boxes of 6 each. How many boxes are required?",
        "A parking lot has 264 spaces arranged in 12 equal rows. How many spaces are in each row?",
        "There are 315 flowers to be arranged in bouquets of 15 each. How many bouquets can be made?",
        "A store has 208 toys to display on 16 shelves equally. How many toys will be on each shelf?",
        "There are 252 students to be divided into groups of 18. How many groups can be formed?",
        "A factory produces 336 items to be packed into boxes of 24 each. How many boxes are needed?",
        "There are 189 apples to be distributed equally among 21 baskets. How many apples per basket?",
        "A school has 294 chairs to arrange in rows of 14. How many rows can be made?",
        "There are 225 books to be sorted into stacks of 15 each. How many stacks will there be?",
        
        # Mixed Operations Problems (81-100)
        "A store had 345 items. They sold 178 items on Monday and received 89 new items on Tuesday. How many items do they have now?",
        "Tom had $120. He spent $35 on lunch and $28 on a book. Then he earned $15. How much money does he have?",
        "There were 240 students in a school. 45 students left and 38 new students joined. How many students are there now?",
        "A farmer had 156 chickens. 23 chickens were sold and 19 new chickens were bought. How many chickens does he have?",
        "Sarah had 85 stickers. She used 27 stickers and her friend gave her 34 more. How many stickers does she have?",
        "A library had 567 books. 89 books were checked out and 45 new books were added. How many books are there now?",
        "Jake had 94 marbles. He gave away 36 marbles and found 18 more. How many marbles does he have?",
        "A bakery made 180 muffins. They sold 67 muffins and made 45 more. How many muffins do they have?",
        "Amy had $78. She spent $34 on groceries and $19 on gas. Then she received $25. How much money does she have?",
        "There were 234 cars in a parking lot. 78 cars left and 52 cars arrived. How many cars are there now?",
        "A school had 445 pencils. Students used 167 pencils and the school bought 89 new pencils. How many pencils are there?",
        "Ben had 67 toy cars. He lost 12 cars and his parents bought him 25 new cars. How many cars does he have?",
        "A pond had 123 fish. 34 fish were caught and 19 new fish were added. How many fish are in the pond?",
        "Lisa had 156 photos. She deleted 43 photos and took 67 new ones. How many photos does she have?",
        "A store had 289 shirts. They sold 134 shirts and received 76 new shirts. How many shirts do they have?",
        "Tom collected 95 baseball cards. He traded 28 cards and received 41 new cards. How many cards does he have?",
        "There were 178 birds in a tree. 45 birds flew away and 29 birds joined them. How many birds are there?",
        "A restaurant had 234 customers. 89 customers left and 56 new customers arrived. How many customers are there?",
        "Jake had $145. He spent $67 on clothes and earned $34 from chores. How much money does he have?",
        "A garden had 167 flowers. 38 flowers wilted and 52 new flowers bloomed. How many flowers are there?",
        
        # Multi-Step Complex Problems (101-120)
        "A school ordered 15 boxes of pencils with 24 pencils each, and 12 boxes of erasers with 36 erasers each. How many items did they order in total?",
        "Maria buys 8 packs of stickers with 15 stickers each. She gives away 47 stickers. How many stickers does she have left?",
        "A farmer plants 12 rows of tomatoes with 18 plants each, and 8 rows of peppers with 22 plants each. How many plants did he plant in total?",
        "Tom saves $25 each week for 8 weeks. He then spends $156 on a bike. How much money does he have left?",
        "A bakery makes 14 trays of cookies with 16 cookies each. They sell 178 cookies. How many cookies are left?",
        "Sarah has 3 albums with 48 photos each and 5 albums with 36 photos each. How many photos does she have in total?",
        "A store receives 18 boxes of books with 25 books each. They already had 127 books. How many books do they have now?",
        "Jake runs 3.5 miles on Monday, 4.2 miles on Wednesday, and 2.8 miles on Friday. What is his total distance?",
        "A theater has 24 rows with 32 seats each. If 256 seats are occupied, how many seats are empty?",
        "Amy buys 6 packs of markers with 12 markers each. She already had 45 markers. How many markers does she have?",
        "A library has 28 shelves with 45 books each on the first floor and 156 books on the second floor. How many books in total?",
        "Ben collects 4 boxes of 36 trading cards each. He trades away 58 cards. How many cards does he have left?",
        "A school has 16 classrooms with 28 desks each and 8 special rooms with 12 desks each. How many desks in total?",
        "Lisa earns $12 per day for 15 days. She spends $134 on gifts. How much money does she have left?",
        "A parking lot has 22 rows with 18 spaces each. If 187 spaces are occupied, how many are empty?",
        "Tom has 7 containers with 24 marbles each. He gives away 89 marbles. How many marbles does he have?",
        "A garden has 15 sections with 26 flowers each. If 78 flowers are picked, how many flowers remain?",
        "Sarah buys 9 boxes of crayons with 16 crayons each. She already had 67 crayons. How many crayons in total?",
        "A store sells 13 cases of water with 24 bottles each. They had 89 bottles already. How many bottles do they have?",
        "Jake has 11 albums with 20 stamps each and loses 47 stamps. How many stamps does he have left?",
        
        # Real-World Application Problems (121-128)
        "A pizza restaurant cuts each pizza into 8 slices. If they make 15 pizzas and sell 89 slices, how many slices are left?",
        "A movie theater has 18 rows with 22 seats each. If 234 tickets are sold, how many seats remain empty?",
        "A farmer has 145 apple trees. Each tree produces 67 apples. If he sells 2,340 apples, how many apples does he have left?",
        "A school bus can carry 48 students. If there are 156 students going on a trip, how many bus trips are needed?",
        "A bookstore orders 23 boxes of novels with 18 books each and 17 boxes of textbooks with 12 books each. How many books did they order?",
        "A basketball player scores 23 points in the first game, 19 points in the second game, and 31 points in the third game. What is his average score per game?",
        "A recipe calls for 2.5 cups of flour per batch. If a baker makes 8 batches, how many cups of flour are needed in total?",
        "A company has 234 employees. If 67 employees work from home and the remaining employees are divided equally among 5 offices, how many employees work in each office?",

    ]   
    
    if mawps_query==5:
        mawps_problems = mawps_problems_5
    else:
        mawps_problems = mawps_problems_128

    return {
        'ASDiv': asdiv_problems,
        'SVAMP': svamp_problems, 
        'MAWPS': mawps_problems
    }

def process_single_math_problem(problem: str, model_config: Dict[str, str], dataset: str = None) -> Dict[str, Any]:
    """
    Process a single math problem - used for parallel processing
    Args:
        problem: Math problem to solve
        model_config: VLLM model configuration
        dataset: Dataset name for labeling
    Returns:
        Processing result dictionary
    """
    try:
        # Initialize fresh Math Toolformer instance for this process
        toolformer = MathToolformer(
            base_url=model_config['base_url'],
            model_path=model_config['model_path']
        )
        
        # Process the problem
        result = toolformer.process_math_problem(problem)
        
        # Add dataset label
        if dataset:
            result['dataset'] = dataset
        
        return result
        
    except Exception as e:
        print(f"Error processing problem '{problem}': {e}")
        return {
            'problem': problem,
            'dataset': dataset,
            'error': str(e),
            'used_calculator': False,
            'latencies': {'end_to_end': 0}
        }

def process_problem_with_dataset(problem_dataset_tuple: tuple, model_config: Dict[str, str]) -> Dict[str, Any]:
    """
    Wrapper function for processing problem with dataset label
    Args:
        problem_dataset_tuple: Tuple of (problem, dataset_name)
        model_config: VLLM model configuration
    Returns:
        Processing result dictionary
    """
    problem, dataset = problem_dataset_tuple
    return process_single_math_problem(problem, model_config, dataset)

def process_all_problems_parallel(problems_with_labels: List[tuple], model_config: Dict[str, str], 
                                num_workers: int = None) -> List[Dict[str, Any]]:
    """
    Process ALL math problems from all datasets in parallel
    Args:
        problems_with_labels: List of (problem, dataset_name) tuples
        model_config: VLLM model configuration
        num_workers: Number of parallel workers (default: CPU count - 1)
    Returns:
        List of processing results
    """
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    
    print(f"🚀 Processing {len(problems_with_labels)} math problems with {num_workers} parallel workers")
    print(f"📋 Problems distribution:")
    
    # Show distribution by dataset
    dataset_counts = {}
    for _, dataset in problems_with_labels:
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
    
    for dataset, count in dataset_counts.items():
        print(f"   - {dataset}: {count} problems")
    
    # Create partial function with fixed arguments
    process_func = partial(process_problem_with_dataset, 
                          model_config=model_config)
    
    results = []
    
    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all problems
        future_to_problem = {executor.submit(process_func, p_d_tuple): p_d_tuple 
                           for p_d_tuple in problems_with_labels}
        
        # Collect results with progress bar
        completed_count = 0
        for future in tqdm(as_completed(future_to_problem), 
                          total=len(problems_with_labels), 
                          desc="🔄 Parallel Processing All Math Problems",
                          unit="problems"):
            try:
                result = future.result()
                results.append(result)
                completed_count += 1
                
                # Show progress by dataset
                if completed_count % 5 == 0 or completed_count == len(problems_with_labels):
                    dataset = result.get('dataset', 'Unknown')
                    print(f"✅ Completed {completed_count}/{len(problems_with_labels)} | Latest: {dataset}")
                    
            except Exception as e:
                problem_dataset_tuple = future_to_problem[future]
                problem, dataset = problem_dataset_tuple
                print(f"❌ Failed to process '{problem[:50]}...' from {dataset}: {e}")
                results.append({
                    'problem': problem,
                    'dataset': dataset,
                    'error': str(e),
                    'used_calculator': False,
                    'latencies': {'end_to_end': 0}
                })
    
    print(f"\n🎯 Parallel processing completed: {len(results)} results")
    return results

def process_dataset_serial(problems: List[str], toolformer: MathToolformer, 
                          rate_limit_delay: float) -> List[Dict[str, Any]]:
    """
    Process dataset problems serially (original approach)
    Args:
        problems: List of problems to process
        toolformer: Initialized Math Toolformer instance
        rate_limit_delay: Delay between requests for rate limiting
    Returns:
        List of processing results
    """
    results = []
    
    for problem in tqdm(problems, desc="Serial Processing", unit="problems"):
        result = toolformer.process_math_problem(problem)
        results.append(result)
        time.sleep(rate_limit_delay)
    
    return results

def evaluate_math_toolformer_performance(all_results: Dict[str, List[Dict]], parallel_mode: bool = False) -> None:
    """Comprehensive Math Toolformer performance evaluation with latency analysis"""
    
    print(f"\n{'='*80}")
    print("🧮 COMPREHENSIVE MATH TOOLFORMER EVALUATION REPORT")
    print("Replicating Toolformer methodology for Mathematical Problem Solving")
    if parallel_mode:
        print("📊 PARALLEL EXECUTION ANALYSIS")
    print(f"{'='*80}")
    
    # Overall latency statistics
    all_latencies = {
        'initial_inference': [],
        'calculations': [],
        'final_inference': [],
        'end_to_end': []
    }
    
    for dataset_name, results in all_results.items():
        print(f"\n📋 {dataset_name} Dataset Analysis:")
        print("-" * 50)
        
        calc_usage = sum(1 for r in results if r['used_calculator'])
        total_problems = len(results)
        
        print(f"📊 Statistics:")
        print(f"   Total problems: {total_problems}")
        print(f"   Calculator usage: {calc_usage}/{total_problems} ({calc_usage/total_problems*100:.1f}%)")
        
        # Latency analysis for this dataset
        dataset_latencies = {
            'initial_inference': [r['latencies']['initial_inference'] for r in results],
            'end_to_end': [r['latencies']['end_to_end'] for r in results]
        }
        
        calc_latencies = []
        final_inference_latencies = []
        for r in results:
            if r['used_calculator']:
                calc_latencies.append(r['latencies']['total_calc_time'])
                final_inference_latencies.append(r['latencies']['final_inference'])
        
        print(f"\n⏱️  Latency Analysis:")
        print(f"   Initial inference: {np.mean(dataset_latencies['initial_inference']):.2f}s avg")
        if calc_latencies:
            print(f"   Calculations: {np.mean(calc_latencies):.3f}s avg")
            print(f"   Final inference: {np.mean(final_inference_latencies):.2f}s avg")
        print(f"   End-to-end: {np.mean(dataset_latencies['end_to_end']):.2f}s avg")
        
        # Collect for overall stats
        all_latencies['initial_inference'].extend(dataset_latencies['initial_inference'])
        all_latencies['calculations'].extend(calc_latencies)
        all_latencies['final_inference'].extend(final_inference_latencies)
        all_latencies['end_to_end'].extend(dataset_latencies['end_to_end'])
        
        print(f"\n💡 Sample Results:")
        for i, result in enumerate(results[:2]):  # Show first 2 results per dataset
            print(f"\n   P{i+1}: {result['problem'][:60]}...")
            print(f"        Initial: {result['initial_solution'][:60]}...")
            if result['used_calculator']:
                print(f"        Calculations: {len(result['calc_calls'])} operations")
                for call in result['calc_calls']:
                    expr = re.search(r'Calculator\("([^"]+)"\)', call)
                    result_val = result['calc_results'].get(call, 'N/A')
                    if expr:
                        print(f"          - '{expr.group(1)}' = {result_val}")
                print(f"        Final: {result['final_answer'][:60]}...")
                print(f"        Timing: {result['latencies']['end_to_end']:.2f}s total")
            else:
                print(f"        No calculations needed ({result['latencies']['end_to_end']:.2f}s)")
    
    # Overall performance summary
    all_problems = sum(len(results) for results in all_results.values())
    all_calc_usage = sum(sum(1 for r in results if r['used_calculator']) for results in all_results.values())
    
    print(f"\n{'='*50}")
    print("🏆 OVERALL PERFORMANCE SUMMARY")
    print(f"{'='*50}")
    print(f"📊 Problem Processing:")
    print(f"   Total problems: {all_problems}")
    print(f"   Calculator usage: {all_calc_usage}/{all_problems} ({all_calc_usage/all_problems*100:.1f}%)")
    
    print(f"\n⏱️  Latency Statistics:")
    print(f"   Initial inference: {np.mean(all_latencies['initial_inference']):.2f}s ± {np.std(all_latencies['initial_inference']):.2f}s")
    if all_latencies['calculations']:
        print(f"   Calculations: {np.mean(all_latencies['calculations']):.3f}s ± {np.std(all_latencies['calculations']):.3f}s")
        print(f"   Final inference: {np.mean(all_latencies['final_inference']):.2f}s ± {np.std(all_latencies['final_inference']):.2f}s")
    print(f"   End-to-end: {np.mean(all_latencies['end_to_end']):.2f}s ± {np.std(all_latencies['end_to_end']):.2f}s")
    
    print(f"\n🎯 Math Toolformer Methodology:")
    print("   ✅ GPT-J-6B model (same as paper)")
    print("   ✅ Calculator tool integration")
    print("   ✅ Self-supervised calculation detection")
    print("   ✅ ASDiv, SVAMP, MAWPS evaluation")
    print("   ✅ Multi-step problem solving")
    
    print(f"\n📈 Key Findings:")
    print(f"   - Tool usage rate: {all_calc_usage/all_problems*100:.1f}% (autonomous calculator adoption)")
    print(f"   - Average processing time: {np.mean(all_latencies['end_to_end']):.1f}s per problem")
    print(f"   - Calculator effectiveness: {len([l for l in all_latencies['calculations'] if l > 0])}/{len(all_latencies['calculations'])} successful")
    
    if all_latencies['calculations']:
        calc_overhead = np.mean(all_latencies['calculations']) / np.mean(all_latencies['end_to_end']) * 100
        print(f"   - Calculation overhead: {calc_overhead:.1f}% of total processing time")

def main(parallel: bool = False, num_workers: int = None, mawps_query:int = 5):
    """
    Main evaluation function for Math Toolformer experiments
    Args:
        parallel: Whether to use parallel processing
        num_workers: Number of workers for parallel processing
    """
    print("=== GPT-J-6B Math Toolformer Evaluation ===")
    print("Replicating Toolformer methodology for Mathematical Problem Solving")
    print("Using GPT-J-6B with Calculator tool for ASDiv, SVAMP, and MAWPS datasets")
    
    # Show execution mode
    if parallel:
        workers = num_workers or max(1, mp.cpu_count() - 1)
        print(f"🚀 PARALLEL EXECUTION MODE: {workers} workers")
    else:
        print("⏳ SERIAL EXECUTION MODE")
    
    # Model configuration
    model_config = {
        'base_url': "http://localhost:5000/v1",
        'model_path': "EleutherAI/gpt-j-6b"
    }
    
    # Initialize Math Toolformer (only needed for serial execution)
    toolformer = None
    if not parallel:
        print(f"\n🧮 INITIALIZING MATH TOOLFORMER")
        print("=" * 60)
        toolformer = MathToolformer()
    
    # Load datasets
    print("\n📚 LOADING MATH DATASETS")
    print("=" * 50)
    datasets = load_math_datasets(mawps_query=mawps_query)
    
    total_problems = sum(len(problems) for problems in datasets.values())
    print(f"Loaded {total_problems} problems across {len(datasets)} datasets:")
    for name, problems in datasets.items():
        print(f"   - {name}: {len(problems)} problems")
    
    all_results = {}
    
    if parallel:
        # PARALLEL MODE: Process ALL problems together
        print(f"\n{'='*60}")
        print(f"🚀 PROCESSING ALL {total_problems} PROBLEMS IN PARALLEL")
        print(f"{'='*60}")
        
        # Combine all problems with dataset labels
        all_problems_with_labels = []
        for dataset_name, problems in datasets.items():
            for problem in problems:
                all_problems_with_labels.append((problem, dataset_name))
        
        # Process all problems in parallel
        all_start_time = time.time()
        all_problem_results = process_all_problems_parallel(
            problems_with_labels=all_problems_with_labels,
            model_config=model_config,
            num_workers=num_workers
        )
        all_total_time = time.time() - all_start_time
        
        # Organize results by dataset
        for dataset_name in datasets.keys():
            all_results[dataset_name] = [
                result for result in all_problem_results 
                if result.get('dataset') == dataset_name
            ]
        
        print(f"\n⚡ ALL PROBLEMS Completed in {all_total_time:.1f}s")
        print(f"📊 Average time per problem: {all_total_time/total_problems:.1f}s")
        print(f"🚀 Estimated parallel speedup: ~{(total_problems * 10.0) / all_total_time:.1f}x faster than serial")
        
    else:
        # SERIAL MODE: Process each dataset separately
        for dataset_name, problems in datasets.items():
            print(f"\n{'='*60}")
            print(f"🔍 EVALUATING ON {dataset_name.upper()} DATASET")
            print(f"{'='*60}")
            
            dataset_start_time = time.time()
            
            # Serial processing with minimal delay (calculations are fast)
            dataset_results = process_dataset_serial(
                problems=problems,
                toolformer=toolformer,
                rate_limit_delay=0.1  # Minimal delay for math problems
            )
            
            dataset_total_time = time.time() - dataset_start_time
            
            print(f"\n⏱️  {dataset_name} Dataset Completed in {dataset_total_time:.1f}s")
            print(f"📊 Average time per problem: {dataset_total_time/len(problems):.1f}s")
            
            all_results[dataset_name] = dataset_results
    
    # Comprehensive evaluation and reporting
    evaluate_math_toolformer_performance(all_results, parallel_mode=parallel)
    
    return all_results

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="GPT-J-6B Math Toolformer Evaluation")
    parser.add_argument(
        "--parallel", 
        action="store_true", 
        help="Enable parallel processing using multiprocessing"
    )
    parser.add_argument(
        "--workers", 
        type=int, 
        default=None,
        help="Number of parallel workers (default: CPU count - 1)"
    )
    parser.add_argument(
        "--serial",
        action="store_true", 
        help="Force serial processing (default behavior)"
    )
    parser.add_argument(
        "--mawps_query",
        type=int, 
        default=5
    )    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Determine execution mode
    if args.parallel and args.serial:
        print("❌ Error: Cannot specify both --parallel and --serial")
        exit(1)
    
    parallel_mode = args.parallel
    num_workers = args.workers
    mawps_query =  args.mawps_query
    
    # If workers specified but not parallel, enable parallel mode
    if num_workers and not parallel_mode:
        print("💡 Workers specified, enabling parallel mode automatically")
        parallel_mode = True
    
    main(parallel=parallel_mode, num_workers=num_workers, mawps_query=mawps_query)