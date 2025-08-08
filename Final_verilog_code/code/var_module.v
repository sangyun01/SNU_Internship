module var_module (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_valid,
    output wire        in_ready,
    input  wire [1023:0] a_in,        // 64 x Q8.8
    output reg  [31:0] ex2,           // Q16.16
    output reg         out_valid,
    input  wire        out_ready
);

    // Pipeline tracker (10-cycle)
    reg [9:0] valid_pipe;
    wire fire_in = in_valid && in_ready;
    wire fire_out = out_valid && out_ready;
    assign in_ready = ~valid_pipe[9];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            valid_pipe <= 10'd0;
        else
            valid_pipe <= {valid_pipe[8:0], fire_in};
    end

    // Unpack input
    wire [15:0] x [0:63];
    wire [31:0] x_sq [0:63];

    genvar i;
    generate
        for (i = 0; i < 64; i = i + 1) begin : MUL
            assign x[i] = a_in[i*16 +: 16];
            mult_gen_mult mul_inst (
                .A(x[i]),
                .B(x[i]),
                .CLK(clk),
                .P(x_sq[i])   // Q16.16
            );
        end
    endgenerate

    // Adder tree for x² sum
    wire [31:0] add_stage1 [0:31];
    wire [31:0] add_stage2 [0:15];
    wire [31:0] add_stage3 [0:7];
    wire [31:0] add_stage4 [0:3];
    wire [31:0] add_stage5 [0:1];
    wire [31:0] sum_sq;

    generate
        for (i = 0; i < 32; i = i + 1)
            c_addsub_add32 add1 (.A(x_sq[2*i]), .B(x_sq[2*i+1]), .CLK(clk), .CE(1'b1), .S(add_stage1[i]));
        for (i = 0; i < 16; i = i + 1)
            c_addsub_add32 add2 (.A(add_stage1[2*i]), .B(add_stage1[2*i+1]), .CLK(clk), .CE(1'b1), .S(add_stage2[i]));
        for (i = 0; i < 8; i = i + 1)
            c_addsub_add32 add3 (.A(add_stage2[2*i]), .B(add_stage2[2*i+1]), .CLK(clk), .CE(1'b1), .S(add_stage3[i]));
        for (i = 0; i < 4; i = i + 1)
            c_addsub_add32 add4 (.A(add_stage3[2*i]), .B(add_stage3[2*i+1]), .CLK(clk), .CE(1'b1), .S(add_stage4[i]));
        for (i = 0; i < 2; i = i + 1)
            c_addsub_add32 add5 (.A(add_stage4[2*i]), .B(add_stage4[2*i+1]), .CLK(clk), .CE(1'b1), .S(add_stage5[i]));
    endgenerate

    c_addsub_add32 final_add (.A(add_stage5[0]), .B(add_stage5[1]), .CLK(clk), .CE(1'b1), .S(sum_sq));

    // Divide by 64 = >>6
    always @(posedge clk) begin
        if (valid_pipe[9]) begin
            ex2 <= sum_sq >> 6;
            out_valid <= 1'b1;
        end else if (fire_out) begin
            out_valid <= 1'b0;
        end
    end

endmodule
