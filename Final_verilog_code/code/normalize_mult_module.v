module normalize_mult_module #(
    parameter N = 64
)(
    input  wire              clk,
    input  wire              rst_n,
    input  wire              in_valid,
    output wire              in_ready,
    input  wire [N*16-1:0]   diff_in,
    input  wire [15:0]       inv_std,
    output wire [N*16-1:0]   norm_out,
    output reg               out_valid,
    input  wire              out_ready
);
    assign in_ready = ~out_valid | out_ready;

    wire [15:0] diff   [0:N-1];
    wire [31:0] prod   [0:N-1];
    wire [15:0] result [0:N-1];

    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : MULT_GEN
            assign diff[i] = diff_in[i*16 +: 16];

            mult_gen_mult mult_inst (
                .A(diff[i]),
                .B(inv_std),
                .CLK(clk),
                .P(prod[i])
            );

            assign result[i] = prod[i][23:8];  // Q16.16 → Q8.8
            assign norm_out[i*16 +: 16] = result[i];
        end
    endgenerate

    // delay line for output valid
    reg [4:0] pipeline;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            pipeline <= 5'b0;
        else
            pipeline <= {pipeline[3:0], (in_valid && in_ready)};
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            out_valid <= 0;
        else if (out_valid && out_ready)
            out_valid <= 0;
        else if (pipeline[4]) // 5th cycle from input
            out_valid <= 1;
    end

endmodule
