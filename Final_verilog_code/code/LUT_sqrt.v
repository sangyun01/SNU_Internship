module LUT_sqrt (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_valid,
    input  wire [15:0] x_in,            // Q8.8 입력값 (분산)
    output reg         out_valid,
    output reg  [15:0] slope_out,
    output reg  [15:0] intercept_out
);

    localparam integer N_SEGMENTS = 8;

    // LUT 정의
    reg [15:0] breakpoints [0:N_SEGMENTS];
    reg [15:0] slopes      [0:N_SEGMENTS-1];
    reg [15:0] intercepts  [0:N_SEGMENTS-1];

    initial begin
        breakpoints[0] = 16'h0003; // 0.0117
        breakpoints[1] = 16'h00F7; // 0.9648
        breakpoints[2] = 16'h04A9; // 4.6602
        breakpoints[3] = 16'h0C03; // 12.0117
        breakpoints[4] = 16'h17BF; // 23.7461
        breakpoints[5] = 16'h28A1; // 40.6328
        breakpoints[6] = 16'h3F38; // 63.2188
        breakpoints[7] = 16'h5BDB; // 91.8555
        breakpoints[8] = 16'h7F00; // 127.0000

        slopes[0] = 16'h00E2; // 0.8828
        slopes[1] = 16'h004F; // 0.3086
        slopes[2] = 16'h002D; // 0.1758
        slopes[3] = 16'h001F; // 0.1211
        slopes[4] = 16'h0017; // 0.0898
        slopes[5] = 16'h0012; // 0.0703
        slopes[6] = 16'h000F; // 0.0586
        slopes[7] = 16'h000C; // 0.0469

        intercepts[0] = 16'h0038; // 0.2188
        intercepts[1] = 16'h00C6; // 0.7734
        intercepts[2] = 16'h0166; // 1.3984
        intercepts[3] = 16'h0214; // 2.0781
        intercepts[4] = 16'h02CE; // 2.8086
        intercepts[5] = 16'h0393; // 3.5742
        intercepts[6] = 16'h0461; // 4.3789
        intercepts[7] = 16'h0536; // 5.2109
    end

    // [1단계] 입력 x_in 등록
    reg [15:0] x_in_reg;
    reg        in_valid_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x_in_reg      <= 16'd0;
            in_valid_reg  <= 1'b0;
        end else begin
            x_in_reg      <= x_in;
            in_valid_reg  <= in_valid;
        end
    end

    // [2단계] 구간 결정 (combinational)
    reg [2:0] next_region;

    always @(*) begin
        if      (x_in_reg < breakpoints[1]) next_region = 0;
        else if (x_in_reg < breakpoints[2]) next_region = 1;
        else if (x_in_reg < breakpoints[3]) next_region = 2;
        else if (x_in_reg < breakpoints[4]) next_region = 3;
        else if (x_in_reg < breakpoints[5]) next_region = 4;
        else if (x_in_reg < breakpoints[6]) next_region = 5;
        else if (x_in_reg < breakpoints[7]) next_region = 6;
        else                                next_region = 7;
    end

    // [3단계] region 등록
    reg [2:0] region;
    reg       region_valid;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            region <= 3'd0;
            region_valid <= 1'b0;
        end else begin
            region       <= next_region;
            region_valid <= in_valid_reg;
        end
    end

    // [4단계] 출력 등록
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            slope_out     <= 16'd0;
            intercept_out <= 16'd0;
            out_valid     <= 1'b0;
        end else begin
            slope_out     <= slopes[region];
            intercept_out <= intercepts[region];
            out_valid     <= region_valid;
        end
    end

endmodule
