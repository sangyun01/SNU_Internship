module mean_module (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_valid,
    output wire        in_ready,
    input  wire [1023:0] a_in,        // 64 x Q8.8
    output reg  [15:0] mean,          // Q8.8
    output reg  [31:0] mean_sq,       // Q16.16
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

    // Input unpack
    wire [15:0] x [0:63];
    genvar i;
    generate
        for (i = 0; i < 64; i = i + 1)
            assign x[i] = a_in[i*16 +: 16];
    endgenerate

    // Stage 0~4: Adder Tree with c_addsub_add (1-cycle)
    wire [15:0] add_stage1 [0:31];
    wire [15:0] add_stage2 [0:15];
    wire [15:0] add_stage3 [0:7];
    wire [15:0] add_stage4 [0:3];
    wire [15:0] add_stage5 [0:1];
    wire [15:0] sum_x;

    generate
        for (i = 0; i < 32; i = i + 1) begin : ADD1
            c_addsub_add add_inst (.A(x[2*i]), .B(x[2*i+1]), .CLK(clk), .CE(1'b1), .S(add_stage1[i]));
        end
        for (i = 0; i < 16; i = i + 1) begin : ADD2
            c_addsub_add add_inst (.A(add_stage1[2*i]), .B(add_stage1[2*i+1]), .CLK(clk), .CE(1'b1), .S(add_stage2[i]));
        end
        for (i = 0; i < 8; i = i + 1) begin : ADD3
            c_addsub_add add_inst (.A(add_stage2[2*i]), .B(add_stage2[2*i+1]), .CLK(clk), .CE(1'b1), .S(add_stage3[i]));
        end
        for (i = 0; i < 4; i = i + 1) begin : ADD4
            c_addsub_add add_inst (.A(add_stage3[2*i]), .B(add_stage3[2*i+1]), .CLK(clk), .CE(1'b1), .S(add_stage4[i]));
        end
        for (i = 0; i < 2; i = i + 1) begin : ADD5
            c_addsub_add add_inst (.A(add_stage4[2*i]), .B(add_stage4[2*i+1]), .CLK(clk), .CE(1'b1), .S(add_stage5[i]));
        end
    endgenerate

    c_addsub_add add_inst_final (.A(add_stage5[0]), .B(add_stage5[1]), .CLK(clk), .CE(1'b1), .S(sum_x));

    // Stage 5~9: mean^2 (Q16.16) using mult_gen_mult (4-cycle)
    wire [15:0] mean_q8 = sum_x >> 6;            // calculate mean
    wire [31:0] mean_sq_out;
    wire        mult_valid;

    mult_gen_mult mult_inst (
        .A(mean_q8),
        .B(mean_q8),
        .CLK(clk),
        .P(mean_sq_out)
    );

    // Output
    always @(posedge clk) begin
        if (valid_pipe[9]) begin
            mean     <= sum_x >> 6;
            mean_sq  <= mean_sq_out;
            out_valid <= 1'b1;
        end else if (fire_out) begin
            out_valid <= 1'b0;
        end
    end

endmodule
