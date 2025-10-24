"""
N-Queens Problem Solver using Grover's Algorithm (Corrected)
- Uses binary encoding: each queen's column is encoded in log2(n) qubits per row.
- Oracle marks valid solutions only (using classical enumeration).
- Diffuser and MCX gates use correct Qiskit syntax.
- Handles bitstring ordering for Qiskit measurement.
"""
import math
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from itertools import permutations
import os

class NQueensGrover:
    def __init__(self, n):
        self.n = n
        self.bits_per_queen = int(math.ceil(math.log2(n))) if n > 1 else 1
        self.num_qubits = n * self.bits_per_queen
        self.solutions = []

    def is_valid_solution(self, board):
        n = len(board)
        if len(set(board)) != n:
            return False
        for i in range(n):
            for j in range(i+1, n):
                if abs(board[i] - board[j]) == abs(i - j):
                    return False
        return True

    def find_classical_solutions(self):
        solutions = []
        for perm in permutations(range(self.n)):
            if self.is_valid_solution(list(perm)):
                solutions.append(list(perm))
        return solutions

    def solution_to_binary(self, solution):
        binary = ""
        for col in solution:
            bits = format(col, f'0{self.bits_per_queen}b')
            binary += bits
        return binary

    def mark_solution(self, circuit, qreg, binary_solution):
        # Apply X gates to qubits that should be 0 in the solution
        for i, bit in enumerate(binary_solution):
            if bit == '0':
                circuit.x(qreg[i])
        # Multi-controlled Z gate using H-MCX-H pattern
        if len(binary_solution) > 1:
            target = qreg[len(binary_solution)-1]
            controls = [qreg[j] for j in range(len(binary_solution)-1)]
            circuit.h(target)
            circuit.mcx(controls, target)
            circuit.h(target)
        else:
            circuit.z(qreg[0])
        # Undo X gates
        for i, bit in enumerate(binary_solution):
            if bit == '0':
                circuit.x(qreg[i])

    def create_oracle(self, qreg):
        oracle = QuantumCircuit(qreg)
        valid_solutions = self.find_classical_solutions()
        for solution in valid_solutions:
            binary = self.solution_to_binary(solution)
            self.mark_solution(oracle, qreg, binary)
        return oracle

    def create_diffuser(self, qreg):
        diffuser = QuantumCircuit(qreg)
        diffuser.h(qreg)
        diffuser.x(qreg)
        # Multi-controlled Z on all qubits
        target = qreg[self.num_qubits-1]
        controls = [qreg[j] for j in range(self.num_qubits-1)]
        diffuser.h(target)
        diffuser.mcx(controls, target)
        diffuser.h(target)
        diffuser.x(qreg)
        diffuser.h(qreg)
        return diffuser

    def decode_quantum_result(self, bitstring):
        # Qiskit returns little-endian bitstrings; reverse for decoding
        bitstring = bitstring[::-1]
        solution = []
        for i in range(self.n):
            start = i * self.bits_per_queen
            end = start + self.bits_per_queen
            col = int(bitstring[start:end], 2)
            solution.append(col)
        return solution

    def solve(self, shots=8192):
        # Smart fallback: Use classical solver for very large n
        if self.n > 8:
            print(f"🧠 n = {self.n} is too large for any quantum simulation. Using classical solver...")
            return self.find_classical_solutions()

        qreg = QuantumRegister(self.num_qubits, 'q')
        creg = ClassicalRegister(self.num_qubits, 'c')
        circuit = QuantumCircuit(qreg, creg)
        circuit.h(qreg)
        oracle = self.create_oracle(qreg)
        diffuser = self.create_diffuser(qreg)
        # Optimal Grover iterations
        num_solutions = len(self.find_classical_solutions())
        N = 2 ** self.num_qubits
        if num_solutions == 0:
            print("No solutions exist for n =", self.n)
            return []
        iterations = int(math.pi/4 * math.sqrt(N / num_solutions))
        iterations = max(1, min(iterations, 6))
        print(f"Grover iterations: {iterations}")
        for _ in range(iterations):
            circuit.compose(oracle, inplace=True)
            circuit.compose(diffuser, inplace=True)
        circuit.measure(qreg, creg)

        # Smart backend selection based on circuit complexity
        use_real_hardware = False
        circuit_complexity = circuit.depth() * circuit.num_qubits
        
        print(f"📊 Circuit complexity score: {circuit_complexity}")
        
        if self.n <= 4 and circuit_complexity < 1000:
            print("✅ Circuit suitable for real quantum hardware")
            use_real_hardware = True
        elif self.n <= 5 and circuit_complexity < 2000:
            print("⚡ Circuit marginal for real quantum hardware - will try but expect noise")
            use_real_hardware = True
        else:
            print("🎯 Circuit too complex for reliable quantum hardware execution")
            print("🔄 Using local AerSimulator for clean quantum algorithm results")
            use_real_hardware = False

        # Try IBM Quantum backends if suitable and API key available
        backend = None
        ibm_api_key = None
        if use_real_hardware and os.path.exists('.env'):
            try:
                with open('.env', 'r') as f:
                    for line in f:
                        if line.strip().startswith('IBM_API_KEY='):
                            ibm_api_key = line.strip().split('=', 1)[1]
                            break
            except Exception:
                ibm_api_key = None

        if use_real_hardware and ibm_api_key:
            try:
                from qiskit_ibm_runtime import QiskitRuntimeService
                # Save/load account for newer Qiskit versions
                QiskitRuntimeService.save_account(channel="ibm_cloud", token=ibm_api_key, overwrite=True)
                service = QiskitRuntimeService(channel="ibm_cloud")
                # Get available backends
                backends = service.backends()
                # Find real IBM quantum hardware
                real_devices = []
                for b in backends:
                    if not b.simulator and b.num_qubits >= self.num_qubits:
                        real_devices.append(b)
                
                if real_devices:
                    # Pick the device with most qubits (usually most capable)
                    backend = max(real_devices, key=lambda x: x.num_qubits)
                    print(f"🚀 Using IBM Quantum REAL hardware: {backend.name} ({backend.num_qubits} qubits)")
                    print("⏳ Note: Real quantum hardware may take several minutes due to queue...")
                else:
                    print("No suitable IBM quantum hardware found.")
                    backend = None
            except Exception as e:
                print("Could not use IBM Quantum backends; falling back to local AerSimulator.")
                print("Reason:", e)

        if backend is None:
            backend = AerSimulator()
            print("Using local AerSimulator backend.")

        # Use IBM Runtime primitives interface for IBM backends, otherwise use backend.run()
        if hasattr(backend, 'provider') and 'IBM' in str(type(backend)):
            from qiskit_ibm_runtime import Sampler
            from qiskit import transpile
            
            print("🔧 Transpiling circuit for IBM quantum hardware...")
            # Transpile circuit for the target backend with high optimization
            transpiled_circuit = transpile(circuit, backend=backend, optimization_level=3)
            print(f"📊 Original circuit: {circuit.depth()} depth, {circuit.num_qubits} qubits")
            print(f"📊 Transpiled circuit: {transpiled_circuit.depth()} depth, {transpiled_circuit.num_qubits} qubits")
            
            # Warn if circuit is very deep (high noise expected)
            if transpiled_circuit.depth() > 50000:
                print("⚠️  WARNING: Very deep circuit detected!")
                print("⚠️  Real quantum hardware will be very noisy for this circuit.")
                print("⚠️  Results may be unreliable due to decoherence.")
                print("💡 Consider using local simulation for n > 4 on current hardware.")
            elif transpiled_circuit.depth() > 10000:
                print("⚠️  CAUTION: Deep circuit - expect some quantum noise in results.")
            
            # Create sampler with backend as mode
            print("🎯 Submitting job to IBM quantum hardware...")
            sampler = Sampler(mode=backend)
            sampler.options.default_shots = shots
            
            # Submit job and wait for results
            job = sampler.run([transpiled_circuit])
            print(f"🎫 Job ID: {job.job_id()}")
            print("⏳ Waiting for quantum execution (this may take several minutes)...")
            
            result = job.result()
            print("✅ Job completed! Processing results...")
            
            # Extract counts from the primitive result (newer API format)
            pub_result = result[0]
            # Try different ways to get measurement data
            if hasattr(pub_result.data, 'meas'):
                counts = pub_result.data.meas.get_counts()
            elif hasattr(pub_result.data, 'c'):
                counts = pub_result.data.c.get_counts()
            else:
                # Fallback: get raw measurement data
                counts = pub_result.data.get_counts()
            print(f"🔬 Raw quantum results: {len(counts)} different measurement outcomes")
        else:
            job = backend.run(circuit, shots=shots)
            result = job.result()
            counts = result.get_counts()
        solutions = []
        seen = set()
        total_shots = sum(counts.values())
        
        # Sort by frequency (most common results first) - these are more likely to be correct
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        
        print(f"🔍 Analyzing {len(sorted_counts)} measurement outcomes...")
        print(f"📊 Top 10 most frequent results:")
        for i, (bitstring, count) in enumerate(sorted_counts[:10]):
            percentage = (count / total_shots) * 100
            print(f"   {i+1}. {bitstring} appeared {count} times ({percentage:.1f}%)")
        
        for bitstring, count in sorted_counts:
            decoded = self.decode_quantum_result(bitstring)
            # Strict filtering: columns within range, all unique, and valid diagonals
            if any((col < 0 or col >= self.n) for col in decoded):
                continue
            if len(set(decoded)) != self.n:
                continue
            if not self.is_valid_solution(decoded):
                continue
            tup = tuple(decoded)
            if tup in seen:
                continue
            
            frequency = count / total_shots
            print(f"✅ Valid solution found: {decoded} (appeared {count} times, {frequency*100:.1f}%)")
            solutions.append(decoded)
            seen.add(tup)
            
            # For noisy quantum hardware, limit to most confident solutions
            if len(solutions) >= 10:  # Stop after finding enough solutions
                break
                
        if len(solutions) == 0:
            print("⚠️  No valid solutions found in quantum results due to noise.")
            print("🔄 This is normal for complex circuits on real quantum hardware.")
            print("💡 For comparison, here are the classical solutions:")
            classical_solutions = self.find_classical_solutions()
            return classical_solutions[:4]  # Return first few classical solutions
            
        return solutions

    def display_board(self, solution):
        n = self.n
        board = [['.']*n for _ in range(n)]
        for row, col in enumerate(solution):
            if 0 <= col < n:
                board[row][col] = 'Q'
        return '\n'.join(''.join(row) for row in board)

if __name__ == "__main__":
    print("N-Queens Grover Solver (Corrected)")
    n = int(input("Enter board size n: "))
    solver = NQueensGrover(n)
    solutions = solver.solve()
    print(f"Found {len(solutions)} unique solutions.")
    for i, sol in enumerate(solutions):
        print(f"Solution {i+1}:")
        print(solver.display_board(sol))
        print()
