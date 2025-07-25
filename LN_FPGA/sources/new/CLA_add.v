module cla_adder #(parameter N = 16)(
    input  [N-1:0] x,
    input  [N-1:0] y,
    input          cin,
    output [N-1:0] sum,
    output         cout
);

// Internal wires
    wire [N-1:0] p, g;
    wire [N:0]   c;

    assign c[0] = cin;

// Compute carry generate & carry propagate function
    generate
        genvar i;
        for (i = 0; i < N; i = i + 1) begin : pq_cla
            assign p[i] = x[i] ^ y[i];
            assign g[i] = x[i] & y[i];
        end
    endgenerate

// Compute the carry for each stage
    generate
        for (i = 1; i < N + 1; i = i + 1) begin : carry_cla
            assign c[i] = g[i-1] | (p[i-1] & c[i-1]);
        end
    endgenerate

// Compute the sum
    generate
        for (i = 0; i < N; i = i + 1) begin : sum_cla
            assign sum[i] = p[i] ^ c[i];
        end
    endgenerate

    assign cout = c[N];

endmodule
