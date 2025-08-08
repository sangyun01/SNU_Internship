module CLA #(parameter N = 16)(
    input  signed [N-1:0] a,
    input  signed [N-1:0] b,
    input          cin,
    output signed [N-1:0] sum,
    output         cout
);

    wire [N-1:0] p, g;
    wire [N:0]   c;

    assign c[0] = cin;

    generate
        genvar i;
        for (i = 0; i < N; i = i + 1) begin : pq_cla
            assign p[i] = a[i] ^ b[i];
            assign g[i] = a[i] & b[i];
        end
    endgenerate

    generate
        for (i = 1; i < N + 1; i = i + 1) begin : carry_cla
            assign c[i] = g[i-1] | (p[i-1] & c[i-1]);
        end
    endgenerate

    generate
        for (i = 0; i < N; i = i + 1) begin : sum_cla
            assign sum[i] = p[i] ^ c[i];
        end
    endgenerate

    assign cout = c[N];

endmodule